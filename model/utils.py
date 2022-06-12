

def make_divisible(val, divisible_by = 2):
    return max(divisible_by, round(val / divisible_by) * divisible_by)


def apply_width_mult(channel, width_mult = 1.0):
    return make_divisible(channel * width_mult)