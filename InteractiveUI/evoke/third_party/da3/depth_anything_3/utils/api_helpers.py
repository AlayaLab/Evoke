import argparse


def parse_scalar(s):
    if not isinstance(s, str):
        return s
    t = s.strip()
    l = t.lower()
    if l == "true":
        return True
    if l == "false":
        return False
    if l in ("none", "null"):
        return None
    try:
        return int(t, 10)
    except Exception:
        pass
    try:
        return float(t)
    except Exception:
        return s


def fn_kv_csv(s: str) -> dict[str, dict[str, object]]:


    result: dict[str, dict[str, object]] = {}
    if not s:
        return result

    for item in s.split(","):
        if not item:
            continue
        parts = item.split(":", 2)
        if len(parts) < 3:
            raise argparse.ArgumentTypeError(f"Bad item '{item}', expected FN:KEY:VALUE")
        fn, key, raw_val = parts[0], parts[1], parts[2]


        if not fn:
            raise argparse.ArgumentTypeError(f"Bad item '{item}': empty function name")
        if not key:
            raise argparse.ArgumentTypeError(f"Bad item '{item}': empty key")

        val = parse_scalar(raw_val)
        bucket = result.setdefault(fn, {})
        bucket[key] = val
    return result
