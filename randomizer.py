from randomtools.tablereader import (
    TableObject, get_global_label, tblpath, addresses, get_random_degree,
    mutate_normal, shuffle_normal)
from randomtools.utils import (
    classproperty, get_snes_palette_transformer,
    read_multi, write_multi, utilrandom as random)
from randomtools.interface import (
    get_outfile, get_seed, get_flags, get_activated_codes,
    run_interface, rewrite_snes_meta, clean_and_write, finish_interface)
from collections import defaultdict
from os import path
from time import time
from collections import Counter


VERSION = 1
ALL_OBJECTS = None
DEBUG_MODE = False


class ItemMixin(object):
    @property
    def rank(self):
        if self.index <= 0:
            return -1

        buy_price, sell_price = (self.old_data["buy_price"],
                                 self.old_data["sell_price"])
        price = max(buy_price, sell_price*2)
        if min(buy_price, sell_price*2) >= 4:
            return price

        return 65536

    @classproperty
    def all_items_and_equipment(self):
        return ItemObject.every + WeaponObject.every + ArmorObject.every

    @classproperty
    def sorted_items_and_equipment(self):
        return sorted(self.all_items_and_equipment,
                      key=lambda i: (i.rank, random.random(), i.pointer))

    def get_similar_all_items(self):
        candidates = self.sorted_items_and_equipment
        candidates = [c for c in candidates if c.rank > 0]
        index = candidates.index(self)
        max_index = len(candidates)-1
        new_index = mutate_normal(index, 0, max_index,
                                  random_degree=self.random_degree)
        return candidates[new_index]


class MagicBitsMixin(object):
    def mutate(self):
        super(MagicBitsMixin, self).mutate()
        self.mutate_magic_bits()

    def mutate_magic_bits(self):
        self.reseed(salt="magicbit")
        for attr in self.magic_bits_attributes:
            value = getattr(self, attr)
            sort_func = lambda x: (
                bin(value ^ x).count('1'), random.random(), value)
            all_values = set([o.old_data[attr] for o in self.every]
                             + [value])
            all_values = sorted(all_values, key=sort_func)
            assert all_values[0] == value
            max_index = len(all_values)-1
            a = random.randint(
                0, random.randint(0, random.randint(0, max_index)))
            b = random.randint(0, max_index)
            if a > b:
                a, b = b, a
            index = int(round((b * self.random_degree) +
                              (a * (1-self.random_degree))))
            new_value = all_values[index]
            assert (value | new_value) == 0xFFFF & (value | new_value)
            new_value = new_value ^ value
            for i in xrange(16):
                mask = (1 << i)
                if random.random() > max(self.random_degree, 0.5):
                    if (new_value & mask):
                        new_value = new_value ^ mask

            value = value ^ new_value
            setattr(self, attr, value)


class EncounterMixin(object):
    @property
    def old_formations(self):
        return ([FormationObject.get(f) for f in self.old_data["common"]] +
            [FormationObject.get(f) for f in self.old_data["uncommon"]] +
            [FormationObject.get(f) for f in self.old_data["rare"]] +
            [FormationObject.get(f) for f in self.old_data["super_rare"]])

    @property
    def rank(self):
        if hasattr(self, "_rank"):
            return self._rank
        if set([f.index for f in self.old_formations]) == set([0]):
            return -1
        formation_ranks = [f.rank for f in self.old_formations]
        weights = [12, 12, 12, 12, 6, 6, 3, 1]
        self._rank = sum([f*w for (f,w) in zip(formation_ranks, weights)])
        return self.rank

    @classmethod
    def intershuffle(cls):
        for e in cls.every:
            if e.rank < 0:
                continue

            for i in xrange(4):
                e2 = e.get_similar(random_degree=cls.random_degree**2)
                chosen = random.choice((e2.old_data["common"]*2)
                                       + e2.old_data["uncommon"])
                e.common[i] = chosen

            for i in xrange(2):
                e2 = e.get_similar(random_degree=cls.random_degree)
                chosen = random.choice(e2.old_data["common"]
                                       + (e2.old_data["uncommon"]*4)
                                       + (e2.old_data["rare"]*4))
                e.uncommon[i] = chosen

            e2 = e.get_similar(random_degree=cls.random_degree**0.5)
            chosen = random.choice(e2.old_data["uncommon"]
                                   + (e2.old_data["rare"]*4)
                                   + (e2.old_data["super_rare"]*2))
            e.rare[0] = chosen

            e2 = e.get_similar(random_degree=cls.random_degree**0.25)
            chosen = random.choice(e2.old_data["rare"]
                                   + (e2.old_data["super_rare"]*2))
            e.super_rare[0] = chosen

    def shuffle(self):
        random.shuffle(self.common)
        random.shuffle(self.uncommon)
        full = self.common + self.uncommon + self.rare + self.super_rare
        full = shuffle_normal(full, random_degree=self.random_degree)
        self.common = full[:4]
        self.uncommon = full[4:6]
        self.rare = full[6:7]
        self.super_rare = full[7:]
        assert full == (self.common + self.uncommon +
                        self.rare + self.super_rare)


class ItemObject(ItemMixin, TableObject):
    mutate_attributes = {
        "buy_price": None,
        "sell_price": None,
    }


class WeaponObject(MagicBitsMixin, ItemMixin, TableObject):
    magic_bits_attributes = ["equipability"]
    mutate_attributes = {
        "attack": None,
        "accuracy": None,
        "evasion": None,
        "strength": None,
        "stamina": None,
        "agility": None,
        "intellect": None,
        "critical_rate": None,
        "hp_boost": None,
        "mp_boost": None,
        "buy_price": None,
        "sell_price": None,
    }
    intershuffle_attributes = [
        "attack", "accuracy", "evasion", "strength", "stamina", "agility",
        "intellect", "critical_rate", "hp_boost", "mp_boost",
        ("buy_price", "sell_price"), "spell_index"]

    @property
    def intershuffle_valid(self):
        return "equipshuffle" in get_activated_codes()


class ArmorObject(MagicBitsMixin, ItemMixin, TableObject):
    magic_bits_attributes = ["equipability"]
    mutate_attributes = {
        "defense": None,
        "weight": None,
        "evasion": None,
        "strength": None,
        "stamina": None,
        "agility": None,
        "intellect": None,
        "hp_boost": None,
        "mp_boost": None,
        "buy_price": None,
        "sell_price": None,
    }
    intershuffle_attributes = [
        "defense", "weight", "evasion", "strength", "stamina", "agility",
        "intellect", "hp_boost", "mp_boost", ("buy_price", "sell_price"),
        "spell_index"]

    @property
    def intershuffle_valid(self):
        return "equipshuffle" in get_activated_codes()


class SpellObject(TableObject):
    mutate_attributes = {
        "accuracy": None,
        "mp_cost": None,
        "price": None,
    }

    @property
    def rank(self):
        return (self.old_data["spell_level"] << 32) | self.old_data["price"]


class SpellClassObject(MagicBitsMixin, TableObject):
    magic_bits_attributes = ["equipability"]


class LevelExpObject(TableObject): pass


class MonsterObject(MagicBitsMixin, TableObject):
    magic_bits_attributes = ["weaknesses", "resistances"]
    mutate_attributes = {
        "exp": None,
        "gil": None,
        "hp": None,
        "morale": None,
        "evasion": None,
        "defense": None,
        "hits": None,
        "accuracy": None,
        "attack": None,
        "agility": None,
        "intellect": None,
        "critical_rate": None,
        "magic_defense": None,
        }
    intershuffle_attributes = [
        "exp", "gil", "hp", "morale", "evasion", "defense", "hits",
        "accuracy", "attack", "agility", "intellect", "critical_rate",
        "status_attack", "magic_defense"]

    @property
    def is_boss(self):
        return 118 <= self.index <= 144 or self.index in [105]

    @property
    def rank(self):
        if hasattr(self, "_rank"):
            return self._rank
        exp_rank = sorted(MonsterObject.every, key=lambda m: m.exp).index(self)
        hp_rank = sorted(MonsterObject.every, key=lambda m: m.hp).index(self)
        self._rank = max(exp_rank, hp_rank)
        return self.rank

    @property
    def drop_type_object(self):
        return {0: None,
                1: ItemObject,
                2: WeaponObject,
                3: ArmorObject}[self.drop_type]

    @property
    def drop(self):
        if self.drop_index <= 0:
            return None
        return self.drop_type_object.get(self.drop_index)

    def set_drop(self, item):
        self.drop_type = {
            ItemObject: 1,
            WeaponObject: 2,
            ArmorObject: 3}[item.__class__]
        self.drop_index = item.index

    def cleanup(self):
        if self.is_boss:
            attrs = sorted(set(
                self.mutate_attributes.keys() + self.intershuffle_attributes
                + ["drop_chance"]))
            for attr in attrs:
                if getattr(self, attr) < self.old_data[attr]:
                    setattr(self, attr, self.old_data[attr])
            for attr in ["morale", "elemental_attack", "status_attack"]:
                setattr(self, attr, self.old_data[attr])
            self.weaknesses &= self.old_data["weaknesses"]
            self.resistances |= self.old_data["resistances"]

        if not self.drop:
            self.drop_chance = 0

        same = self.weaknesses & self.resistances
        self.weaknesses ^= same
        assert not self.weaknesses & self.resistances

    def mutate(self):
        super(MonsterObject, self).mutate()

        self.reseed(salt="extra")
        drop_partner = self.get_similar()
        if drop_partner.old_data["drop_type"] or not self.is_boss:
            for attr in ["drop_type", "drop_index", "drop_chance"]:
                setattr(self, attr, drop_partner.old_data[attr])
            if self.drop:
                new_item = self.drop.get_similar_all_items()
                self.set_drop(new_item)
                self.drop_chance = mutate_normal(
                    self.drop_chance, 0, 100, wide=False,
                    random_degree=self.random_degree)

        self.mutate_magic_bits()


class ShopPointerObject(TableObject): pass


class BaseStatsObject(TableObject):
    mutate_attributes = {
        "hp": None,
        "mp": None,
        "strength": None,
        "agility": None,
        "intellect": None,
        "stamina": None,
        "luck": None,
        "accuracy": None,
        "evasion": None,
        "magic_defense": None,
        }


class MapDataObject(TableObject): pass
class EncPackDistObject(TableObject): pass
class OverworldEncObject(EncounterMixin, TableObject): pass
class DungeonEncObject(EncounterMixin, TableObject): pass

class ChestObject(TableObject):
    @property
    def item_type_object(self):
        if self.is_money:
            return None
        return {0: None,
                1: ItemObject,
                2: WeaponObject,
                3: ArmorObject}[self.item_type]

    @property
    def item(self):
        if self.item_type_object is None:
            return None
        return self.item_type_object.get(self.contents)

    @property
    def is_money(self):
        return not self.get_bit("contains_item")

    @property
    def money_amount(self):
        return self.item_type | (self.contents << 8)

    def set_money_amount(self, value):
        self.set_bit("contains_item", False)
        self.item_type = value & 0xFF
        self.contents = value >> 8

    def set_item(self, item):
        self.set_bit("contains_item", True)
        self.item_type = {
            ItemObject: 1,
            WeaponObject: 2,
            ArmorObject: 3}[item.__class__]
        self.contents = item.index

    @property
    def rank(self):
        if self.is_money:
            return self.money_amount
        if self.item:
            return self.item.rank
        return -1

    def mutate(self):
        if self.rank <= 0 and self.contents:
            return

        self.reseed(salt="mut")
        partner = random.choice(
            [c for c in self.every if c.rank > 0 or c.contents == 0])
        if partner.rank <= 0:
            self.item_type = 0
            self.contents = 0
            self.set_bit("contains_item", True)
            return

        value = self.rank
        if value <= 0:
            high = random.randint(0, partner.rank)
            low = random.randint(0, high)
            value = int(round((high * self.random_degree) +
                              (low * (1-self.random_degree))))

        if not partner.get_bit("contains_item"):
            value = random.randint(int(round(value ** 0.9)), value)
            value = mutate_normal(value, 0, 65536,
                                  random_degree=self.random_degree)
            value = min(value, 65535)
            self.set_money_amount(value)
        else:
            partner = partner.item
            candidates = [c for c in partner.ranked if c.rank <= value]
            if not candidates:
                chosen = partner.ranked[0]
            else:
                chosen = candidates[-1]
            chosen = chosen.get_similar()
            self.set_item(chosen)


class MonsterAIObject(TableObject):
    @property
    def monster(self):
        return MonsterObject.get(self.index)

    @property
    def rank(self):
        return self.monster.rank

    def mutate(self):
        self.ai_index = self.get_similar().old_data["ai_index"]

    def cleanup(self):
        if self.ai_index == 0x2C or self.monster.is_boss:
            self.ai_index = self.old_data["ai_index"]


class AIObject(TableObject):
    mutate_attributes = {
        "spell_cast_chance": None,
        "skill_cast_chance": None,
        }

    def shuffle(self):
        spells = [s for s in self.spell_queue if s != 0xFF]
        if spells:
            length = random.randint(len(set(spells)), 8)
            temp = sorted(set(spells))
            while len(temp) < length:
                temp.append(random.choice(spells))
            random.shuffle(temp)
            self.spell_queue = temp

        skills = [s for s in self.skill_queue if s != 0xFF]
        if skills:
            length = random.randint(len(set(skills)), 4)
            temp = sorted(set(skills))
            while len(temp) < length:
                temp.append(random.choice(skills))
            random.shuffle(temp)
            self.skill_queue = temp

    def cleanup(self):
        if set(self.skill_queue) == set([0xFF]):
            self.skill_cast_chance = 0
            self.spell_cast_chance = max(
                self.spell_cast_chance, self.old_data["spell_cast_chance"])
        else:
            self.skill_cast_chance = max(
                self.skill_cast_chance, self.old_data["skill_cast_chance"])

        if set(self.spell_queue) == set([0xFF]):
            self.spell_cast_chance = 0

        while len(self.spell_queue) < 8:
            self.spell_queue.append(0xFF)
        while len(self.skill_queue) < 4:
            self.skill_queue.append(0xFF)
        assert len(self.spell_queue) == 8
        assert len(self.skill_queue) == 4


class MonsterSizeObject(TableObject): pass


class LevelUpObject(TableObject):
    @classproperty
    def after_order(self):
        return [BaseStatsObject]

    @property
    def class_index(self):
        return self.index / 99

    @property
    def base_stats(self):
        return BaseStatsObject.get(self.class_index)

    @classmethod
    def mutate_stat_curve(cls, class_index, attr):
        lus = [lu for lu in LevelUpObject.every
               if lu.class_index == class_index]
        assert len(lus) == 99
        lus = lus[:98]
        assert len(lus) == 98

        lus[0].reseed(salt="fullmut"+attr)
        bits = [lu.get_bit(attr) for lu in lus]
        value = len([b for b in bits if b])
        base_ratio = value / float(len(lus))
        max_ratio = max([cls.get_class_stat_score(i, attr) / float(len(lus))
                         for i in xrange(6)])
        assert max_ratio >= base_ratio
        base_ratio = mutate_normal(base_ratio, 0, max_ratio, wide=False,
                                   random_degree=LevelUpObject.random_degree,
                                   return_float=True)
        remaining = list(lus)
        while remaining:
            ratio = mutate_normal(base_ratio, 0, max_ratio, wide=False,
                                  random_degree=LevelUpObject.random_degree,
                                  return_float=True)
            max_index = len(remaining)-1
            divider = random.randint(0, max_index)
            aa = remaining[:divider]
            bb = remaining[divider:]

            if len(aa) > len(bb):
                aa, bb = bb, aa
            elif len(aa) == len(bb) and random.choice([True, False]):
                aa, bb = bb, aa
            if random.choice([True, True, False]):
                to_set, remaining = aa, bb
            else:
                to_set, remaining = bb, aa

            assert len(to_set + remaining) == max_index + 1
            for lu in to_set:
                value = (random.random() < ratio)
                lu.set_bit(attr, value)

    @classmethod
    def get_class_stat_score(cls, class_index, attr):
        lus = [lu for lu in LevelUpObject.every
               if lu.class_index == class_index]
        return len([lu for lu in lus if lu.get_bit(attr)])

    @classmethod
    def full_randomize(cls):
        for class_index in range(6):
            for attr in ["hp", "mp", "strength", "agility", "intellect",
                         "stamina", "luck", "spell_level"]:
                cls.mutate_stat_curve(class_index, attr)
        cls.randomized = True


class LevelAccuracyObject(TableObject): pass
class LevelMagResObject(TableObject): pass
class ItemSpellObject(TableObject): pass


class FormationObject(TableObject):
    mutate_attributes = {"ambush_rate": None}

    @property
    def enemy_types(self):
        enemies = set([])
        for i in xrange(1, 5):
            index = getattr(self, "group%s_index" % i)
            if index == 0xFF:
                continue
            lower = getattr(self, "group%s_min" % i)
            if lower <= 0:
                continue
            m = MonsterObject.get(index)
            enemies.add(m)
        return sorted(enemies, key=lambda m: m.index)

    @property
    def rank(self):
        ranks = [m.rank for m in self.enemy_types]
        return max(ranks)


if __name__ == "__main__":
    try:
        print ("You are using the Final Fantasy Dawn of Souls "
               "randomizer version %s." % VERSION)
        print

        ALL_OBJECTS = [g for g in globals().values()
                       if isinstance(g, type) and issubclass(g, TableObject)
                       and g not in [TableObject]]

        codes = {
            "equipshuffle": ["equipshuffle"],
        }
        run_interface(ALL_OBJECTS, snes=False, codes=codes, custom_degree=True)
        hexify = lambda x: "{0:0>2}".format("%x" % x)
        numify = lambda x: "{0: >3}".format(x)
        minmax = lambda x: (min(x), max(x))

        clean_and_write(ALL_OBJECTS)
        finish_interface()

    except Exception, e:
        print "ERROR: %s" % e
        raw_input("Press Enter to close this program.")
