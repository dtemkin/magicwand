from collections import Counter


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
    if type(items) is dict or type(items) == Counter:
        tot = sum(items.values())
        cnt = items
    elif type(items) is list or type(items) is tuple or type(items) is set:
        items = list(items)
        tot = len(items)
        cnt = Counter(items)
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

