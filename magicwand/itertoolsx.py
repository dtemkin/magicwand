import collections
import functools

class memoized(object):
    # Thanks to @delton137 for providing this function!
    # Source: http://www.moreisdifferent.com/2016/02/08/recursion-is-slow/

   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   Taken from the python decorator library: https://wiki.python.org/moin/PythonDecoratorLibrary#Memoize
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}
      self.__name__ = func.__name__
      self.func_name = func.func_name #python2 support

   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value

   def __repr__(self):
      '''Return the function's docstring.'''
      return self.func.__doc__

   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)


def flatten_nested_list(lst):
    x = []
    for itm in lst:
        if hasattr(itm, "__iter__"):
            x.extend(list(itm))
        else:
            x.append(itm)
        
    return x

def relative_freq(items, float_prec=None, desc=True,
                  is_sorted=True, sort_by='value', as_=dict):
    if type(items) is dict or type(items) == collections.Counter:
        tot = sum(items.values())
        cnt = items
    elif type(items) is list or type(items) is tuple or type(items) is set:
        items = list(items)
        tot = len(items)
        cnt = collections.Counter(items)
    else:
        raise TypeError("Invalid 'items' argument. "
                        "Must be iterable and not %s type." % type(items))

    if float_prec is None:
        itms = [(k, float(cnt[k]/tot)) for k in cnt]
    else:
        itms = [(k, round(cnt[k]/tot, float_prec)) for k in cnt]

    if is_sorted:
        if str(sort_by).lower() in ['value', 'v', '1']:
            itms = sorted(itms, key=lambda x: x[1], reverse=desc)
        elif str(sort_by).lower() in ['key', 'k', '0']:
            itms = sorted(itms, key=lambda x: x[0], reverse=desc)
        else:
            raise ValueError("invalid sort_by value. must be 'value' or 'key'")
    if callable(as_):
        return as_.__call__(itms)


def partition_string(seq, chunk_size, skip_tail=False):
    lst = []
    if chunk_size <= len(seq):
        lst.extend([seq[:chunk_size]])
        lst.extend(partition_string(seq[chunk_size:], chunk_size, skip_tail))
    elif not skip_tail and seq:
        lst.extend([seq])
    return lst

