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


class ItemObject(TableObject): pass
class WeaponObject(TableObject): pass
class ArmorObject(TableObject): pass
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
class ChestObject(TableObject): pass
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
