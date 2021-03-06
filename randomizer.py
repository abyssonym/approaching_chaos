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


VERSION = 2
ALL_OBJECTS = None
DEBUG_MODE = False


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
    flag = 'e'
    custom_random_enable = True

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


class PriceMixin(object):
    def price_clean(self):
        power = 0
        price = self.price
        if price == 0:
            return

        price = price * 2
        while 0 < price < 100000:
            price *= 10
            power += 1
        price = int(round(price, -4))
        price /= (10**power)
        price = price / 2
        if price > 10 and price % 10 == 0 and VERSION % 2 == 1:
            price = price - 1
        price = min(price, 99999)

        if hasattr(self, "buy_price"):
            self.buy_price = price
            if self.sell_price > 0:
                self.sell_price = min(self.sell_price, price / 2)
        else:
            self.price = price

    def cleanup(self):
        self.price_clean()


class ItemMixin(PriceMixin):
    flag = 'q'
    custom_random_enable = True

    @property
    def rank(self):
        if self.index <= 0:
            return -1

        buy_price, sell_price = (self.old_data["buy_price"],
                                 self.old_data["sell_price"])
        price = max(buy_price, sell_price*2)
        if min(buy_price, sell_price*2) >= 4:
            return price

        if self.__class__ is ItemObject:
            return 65536
        else:
            return 65537

    @property
    def price(self):
        return self.buy_price

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

    def cleanup(self):
        self.price_clean()

        if self.index == 0:
            for attr in self.old_data.keys():
                setattr(self, attr, self.old_data[attr])

        for attr in self.old_data.keys():
            if attr in ["attack", "defense"] and getattr(self, attr) == 0:
                setattr(self, attr, self.old_data[attr])


class ItemObject(ItemMixin, TableObject):
    flag_description = "items and equipability"

    mutate_attributes = {
        "buy_price": None,
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


class SpellObject(PriceMixin, TableObject):
    flag = 's'
    flag_description = "spells and spell equipability"
    custom_random_enable = True

    mutate_attributes = {
        #"accuracy": None,
        "mp_cost": None,
        "price": None,
    }

    @property
    def rank(self):
        return (self.old_data["spell_level"] << 32) | self.old_data["price"]


class SpellClassObject(MagicBitsMixin, TableObject):
    flag = 's'
    custom_random_enable = True

    magic_bits_attributes = ["equipability"]


class LevelExpObject(TableObject): pass


class MonsterObject(MagicBitsMixin, TableObject):
    flag = 'm'
    flag_description = "monster stats and drops"
    custom_random_enable = True

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


class ShopPointerObject(TableObject):
    flag = 'p'
    flag_description = "shops"
    custom_random_enable = True

    @classproperty
    def after_order(self):
        return [WeaponObject, ArmorObject, ItemObject]

    def read_data(self, filename, pointer=None):
        super(ShopPointerObject, self).read_data(filename, pointer)
        assert self.zero == 0
        assert self.eight == 8
        f = open(filename, "r+b")
        self.wares_pointer = self.shop_pointer
        if 1 <= self.shop_type <= 3:
            self.wares_pointer += 1
        f.seek(self.wares_pointer)
        self.wares_indexes = map(ord, f.read(self.num_items))
        f.close()
        self.rank

    def write_data(self, filename, pointer=None, syncing=True):
        super(ShopPointerObject, self).write_data(
            filename, pointer, syncing=syncing)
        assert len(self.wares_indexes) == self.num_items
        f = open(filename, "r+b")
        f.seek(self.wares_pointer)
        f.write("".join(map(chr, self.wares_indexes)))
        f.close()

    @property
    def shop_type(self):
        return self.type_and_number >> 4

    @property
    def num_items(self):
        return self.type_and_number & 0xF

    @property
    def item_type_object(self):
        return {1: WeaponObject,
                2: ArmorObject,
                3: ItemObject,
                4: ItemObject,  # Caravan
                5: SpellObject,
                6: SpellObject}[self.shop_type]

    @property
    def valid_wares(self):
        assert self.shop_type > 0
        if self.shop_type <= 4:
            return [c for c in self.item_type_object.every if c.rank >= 0]
        elif self.shop_type == 5:
            return [s for s in SpellObject.every if 1 <= s.index <= 0x20]
        elif self.shop_type == 6:
            return [s for s in SpellObject.every if s.index >= 0x21]

    @property
    def wares(self):
        return [self.item_type_object.get(i) for i in self.wares_indexes]

    @property
    def rank(self):
        if hasattr(self, "_rank"):
            return self._rank
        try:
            prices = [w.price for w in self.wares]
        except KeyError, e:
            if get_global_label() == "FFDOS_MOB":
                self._rank = -1
                return self.rank
            else:
                raise e
        self._rank = int(round(sum(prices) / len(prices)))
        return self.rank

    @classmethod
    def intershuffle(cls):
        white_shops = [s for s in cls.ranked if s.shop_type == 5]
        black_shops = [s for s in cls.ranked if s.shop_type == 6]
        for shops in [white_shops, black_shops]:
            wares = []
            for s in shops:
                s.reseed(salt="spell_shuffle")
                shop_wares = s.wares_indexes
                random.shuffle(shop_wares)
                wares.extend(shop_wares)
            wares = shuffle_normal(wares, random_degree=cls.random_degree)
            for s in shops:
                assert len(s.wares_indexes) <= len(wares)
                s.wares_indexes = wares[:len(s.wares_indexes)]
                wares = wares[len(s.wares_indexes):]
            assert len(wares) == 0

    def mutate(self):
        if self.shop_type >= 5:
            return
        if self.index == 0x26:
            assert self.shop_type == 4
            assert len(self.wares) == 1
            return
        if self.rank < 0:
            return
        new_wares = []
        candidates = [c for c in self.valid_wares
                      if c.old_data["buy_price"] > 2]
        wares = self.wares
        random.shuffle(wares)
        for w in wares:
            while True:
                nw = w.get_similar(candidates,
                                   random_degree=self.random_degree)
                if nw in new_wares:
                    nw = random.choice(self.valid_wares)
                if nw not in new_wares:
                    break
            if nw.rank >= 65536:
                price = max(30000, nw.buy_price, nw.sell_price*2)
                if price < 50000:
                    price *= 2
                price = min(price, 99999)
                price = mutate_normal(price, 0, 99999, wide=True,
                                      random_degree=nw.random_degree)
                nw.buy_price = price
            new_wares.append(nw)
        self.wares_indexes = sorted([nw.index for nw in new_wares])


class BaseStatsObject(TableObject):
    flag = 'c'
    flag_description = "character class stats"
    custom_random_enable = True

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
    flag = 't'
    flag_description = "treasure chests"
    custom_random_enable = True

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
            [c for c in self.every if
             c.rank > 0 and c.item_type > 0 and c.contents > 0])

        value = self.rank
        if value <= 0:
            high = random.randint(0, partner.rank)
            low = random.randint(0, high)
            value = int(round((high * self.random_degree) +
                              (low * (1-self.random_degree))))

        if not partner.get_bit("contains_item"):
            value = random.randint(int(round(value ** 0.9)), value)
            value = min(value, 65537)
            value = mutate_normal(value, 1, 65537,
                                  random_degree=self.random_degree)
            value = min(value, 65535)
            self.set_money_amount(value)
        else:
            partner = partner.item
            candidates = [c for c in partner.ranked if 0 < c.rank <= value]
            if not candidates:
                candidates = [c for c in partner.ranked if c.rank > 0]
                chosen = candidates[0]
            else:
                chosen = candidates[-1]
            chosen = chosen.get_similar(
                random_degree=ChestObject.random_degree)
            assert chosen.index > 0
            self.set_item(chosen)


class MonsterAIObject(TableObject):
    flag = 'a'
    flag_description = "monster AI"
    custom_random_enable = True

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
    flag = 'a'
    custom_random_enable = True

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
    flag = 'c'
    custom_random_enable = True

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
    def mutate_spell_levels(cls, class_index):
        lus = [lu for lu in LevelUpObject.every
               if lu.class_index == class_index]
        indexes = [i for (i, lu) in enumerate(lus)
                   if lu.get_bit("spell_level")]
        if not indexes:
            return

        new_indexes = []
        for i in indexes:
            while True:
                n = mutate_normal(i, 1, 40, random_degree=cls.random_degree)
                if n not in new_indexes:
                    new_indexes.append(n)
                    break

        if len(new_indexes) > 1:
            remaining = 8 - len(new_indexes)
            highest = max(new_indexes)
            new_indexes.extend(random.sample(range(highest+1, 50), remaining))
            assert len(new_indexes) == 8

        assert len(new_indexes) in [1, 8]
        for i, lu in enumerate(lus):
            if i in new_indexes:
                lu.set_bit("spell_level", True)
            else:
                lu.set_bit("spell_level", False)

    @classmethod
    def full_randomize(cls):
        for class_index in range(6):
            cls.class_reseed(salt="stats%s" % class_index)
            for attr in ["hp", "mp", "strength", "agility", "intellect",
                         "stamina", "luck"]:
                cls.mutate_stat_curve(class_index, attr)
            cls.class_reseed(salt="spells%s" % class_index)
            cls.mutate_spell_levels(class_index)
        cls.randomized = True


class LevelAccuracyObject(TableObject):
    flag = 'c'
    mutate_attributes = {"accuracy": None}


class LevelMagResObject(TableObject):
    flag = 'c'
    mutate_attributes = {"mag_res": None}


class ItemSpellObject(TableObject): pass


class FormationObject(TableObject):
    flag = 'e'
    flag_description = "encounters"
    custom_random_enable = True

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


class FF2ChestObject(TableObject):
    flag = 't'
    custom_random_enable = True

    def mutate(self):
        if self.misc in [2, 4] and self.contents < 0x10:
            return
        partner = random.choice(FF2ChestObject.every)
        assert self.misc in [1, 2, 4]
        assert partner.misc in [1, 2, 4]
        if partner.misc == 1:
            if self.misc == 1:
                value = self.contents
            else:
                value = ff2_get_price(self.contents)
            self.misc = 1
            value = min(65000, max(0, value))
            self.contents = mutate_normal(value, 0, 65000, wide=True,
                                          random_degree=self.random_degree)
        else:
            if self.misc == 1:
                item = ff2_get_item_valued(self.contents)
                self.misc = 2
            else:
                item = self.contents
            item = ff2_get_similar_item(item, random_degree=self.random_degree)
            self.contents = item
        assert self.misc in [1, 2, 4]


class FF2GilValueObject(TableObject): pass


class FF2MonsterObject(MagicBitsMixin, TableObject):
    flag = 'm'
    custom_random_enable = True

    magic_bits_attributes = ["resistances", "weaknesses", "absorbs"]
    mutate_attributes = {
        "hp": None,
        "mp": None,
        "accuracy_low": None,
        "accuracy_high": None,
        "attack": None,
        "evasion_low": None,
        "evasion_high": None,
        "defense": None,
        "mdef_low": None,
        "mdef_high": None,
    }
    intershuffle_attributes = [
        "hp", "mp", ("accuracy_low", "accuracy_high"), "attack",
        ("evasion_low", "evasion_high"), "defense", ("mdef_low", "mdef_high"),
        "drop_gil_index"]

    @property
    def monster_rank(self):
        return FF2MonsterRankObject.get(self.index).monster_rank

    @property
    def is_boss(self):
        return self.monster_rank >= 7 or self.old_data["hp"] >= 15000

    @property
    def rank(self):
        if hasattr(self, "_rank"):
            return self._rank
        monster_rank = FF2MonsterRankObject.get(
            self.index).old_data["monster_rank"]
        if self.old_data["hp"] >= 15000:
            monster_rank = max(monster_rank, 6.5)
        monster_rank = monster_rank ** 0.5
        self._rank = int(round(self.old_data["hp"] * monster_rank))
        return self.rank

    def mutate(self):
        super(FF2MonsterObject, self).mutate()
        partner = self.get_similar()
        self.drop_gil_index = partner.old_data["drop_gil_index"]

    def cleanup(self):
        if self.is_boss:
            for attr in ["hp", "mp", "accuracy_low", "accuracy_high", "attack",
                         "evasion_low", "evasion_high", "defense", "mdef_low",
                         "mdef_high"]:
                if self.old_data[attr] > getattr(self, attr):
                    setattr(self, attr, self.old_data[attr])
            self.drop_gil_index = self.old_data["drop_gil_index"]
            self.absorbs |= self.old_data["absorbs"]
            self.resistances |= self.old_data["resistances"]
            self.weaknesses &= self.old_data["weaknesses"]

        for attr in ["accuracy", "evasion", "mdef"]:
            lower = "%s_low" % attr
            upper = "%s_high" % attr
            if getattr(self, lower) > getattr(self, upper):
                low = getattr(self, lower)
                setattr(self, lower, getattr(self, upper))
                setattr(self, upper, low)
                assert getattr(self, lower) <= getattr(self, upper)

        resorbs = self.resistances & self.absorbs
        self.resistances ^= resorbs
        weakres = self.weaknesses & (self.resistances | self.absorbs)
        self.weaknesses ^= weakres


class FF2MonsterRankObject(TableObject): pass


class FF2MonsterDropObject(TableObject):
    flag = 't'
    custom_random_enable = True

    def mutate(self):
        for i, di in enumerate(self.drop_indexes):
            if di and not di & 0x80:
                self.drop_indexes[i] = ff2_get_similar_item(
                    di, random_degree=self.random_degree)


class FF2ArmorObject(TableObject): pass


class FF2ShopObject(TableObject):
    flag = 'p'
    custom_random_enable = True

    def mutate(self):
        consumables = set(range(16, 47))
        weapons = set(range(58, 112))
        weapons |= set([192, 193, 194])
        armors = set(range(49, 58)) | set(range(112, 152))
        armors.add(195)
        tomes = set(range(152, 192))
        for itemset in [consumables, weapons, armors, tomes]:
            if set(self.wares) <= itemset:
                chosen = itemset
                break
        else:
            chosen = consumables | weapons | armors | tomes

        wares = list(self.wares)
        random.shuffle(wares)
        new_wares = []
        for w in wares:
            while True:
                new = ff2_get_similar_item(w, chosen,
                                           random_degree=self.random_degree)
                if new not in new_wares:
                    new_wares.append(new)
                    break
        assert len(new_wares) == len(wares)
        self.wares = sorted(new_wares)

        new_prices = []
        for w in self.wares:
            prices = sorted(FF2PriceObject.every,
                            key=lambda po: abs(po.price - ff2_get_price(w)))
            chosen = prices[0]
            temp = chosen.get_similar()
            if temp.price >= (ff2_get_price(w) / 2):
                chosen = temp
            new_prices.append(chosen.index)
        assert len(new_prices) == len(self.prices)
        self.prices = new_prices


class FF2PriceObject(PriceMixin, TableObject):
    flag = 'p'
    custom_random_enable = True

    @property
    def rank(self):
        return self.price


class FF2ItemPriceObject(TableObject): pass
class FF2EquipPriceObject(TableObject): pass


def ff2_get_price(index):
    if 191 <= index <= 195:
        return 65000
    if 0x10 <= index <= 0x30:
        return FF2ItemPriceObject.get(index-0x10).price * 2
    elif 0x31 <= index:
        return FF2EquipPriceObject.get(index-0x31).price * 2
    return None


def ff2_get_ranked_items(candidates=None):
    if candidates is None:
        candidates = range(0x10, 196)
    banned = range(0x10) + [47, 48]
    candidates = [c for c in candidates if c not in banned]
    candidates = sorted(candidates,
                        key=lambda c: (ff2_get_price(c), random.random(), c))
    return candidates


def ff2_get_similar_item(item, candidates=None, random_degree=None):
    candidates = ff2_get_ranked_items(candidates)
    if random_degree is None:
        random_degree = FF2ChestObject.random_degree
    index = candidates.index(item)
    max_index = len(candidates)-1
    new_index = mutate_normal(index, 0, max_index, wide=True,
                              random_degree=random_degree)
    return candidates[new_index]


def ff2_get_item_valued(value):
    candidates = ff2_get_ranked_items()
    lowest = candidates[0]
    candidates = [c for c in candidates if ff2_get_price(c) <= value]
    if not candidates:
        return lowest
    return candidates[-1]


if __name__ == "__main__":
    try:
        print ("You are using the Final Fantasy: Dawn of Souls "
               "APPROACHING CHAOS randomizer version %s." % VERSION)
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

        if DEBUG_MODE:
            for objtype in [ItemObject, WeaponObject,
                            ArmorObject, SpellObject]:
                for o in objtype.every:
                    if hasattr(o, "buy_price"):
                        o.buy_price = 1
                    else:
                        o.price = 1
            for m in MonsterObject.every:
                for attr in ["hp", "evasion", "defense", "hits", "accuracy",
                             "attack", "agility", "intellect", "critical_rate",
                             "magic_defense"]:
                    setattr(m, attr, 1)
                m.exp = 65000
            for l in LevelUpObject.every:
                l.set_bit("spell_level", True)
            for b in BaseStatsObject.every:
                b.spell_level = 8
            for s in SpellObject.every:
                s.mp_cost = 1
            for s in SpellClassObject.every:
                s.equipability = 0x0FFF

        clean_and_write(ALL_OBJECTS)
        finish_interface()

    except Exception, e:
        print "ERROR: %s" % e
        raw_input("Press Enter to close this program.")
