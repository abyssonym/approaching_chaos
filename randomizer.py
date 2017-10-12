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
        price = max(self.buy_price, self.sell_price*2)
        if min(self.buy_price, self.sell_price*2) >= 4:
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
        index = candidates.index(self)
        max_index = len(candidates)-1
        new_index = mutate_normal(index, 0, max_index,
                                  random_degree=self.random_degree)
        return candidates[new_index]

class ItemObject(ItemMixin, TableObject): pass
class WeaponObject(ItemMixin, TableObject): pass
class ArmorObject(ItemMixin, TableObject): pass

class SpellObject(TableObject): pass
class SpellClassObject(TableObject): pass
class LevelExpObject(TableObject): pass
class MonsterObject(TableObject): pass
class ShopPointerObject(TableObject): pass
class BaseStatsObject(TableObject): pass
class MapDataObject(TableObject): pass
class EncPackDistObject(TableObject): pass
class OverworldEncObject(TableObject): pass
class DungeonEncObject(TableObject): pass

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

class MonsterAIObject(TableObject): pass
class AIObject(TableObject): pass
class MonsterSizeObject(TableObject): pass
class LevelUpObject(TableObject): pass
class LevelAccuracyObject(TableObject): pass
class LevelMagResObject(TableObject): pass
class ItemSpellObject(TableObject): pass


if __name__ == "__main__":
    try:
        print ("You are using the Final Fantasy Dawn of Souls "
               "randomizer version %s." % VERSION)
        print

        ALL_OBJECTS = [g for g in globals().values()
                       if isinstance(g, type) and issubclass(g, TableObject)
                       and g not in [TableObject]]

        codes = {
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
