import math

def remove_missing_values(values):
    output = []
    for value in values:
        if value is None:
            continue
        if value == "":
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        output.append(value)
    return output


def fill_missing_values(values, fill_val=0):
    output = []
    for value in values:
        try:
            if math.isnan(value):
                output.append(fill_val)
                continue
        except TypeError:
            pass
        if value is None or value == "":
            output.append(fill_val)
        else:
            output.append(value)
    return output


def unique_values(values):
    output =[] 
    for value in values:
        if value not in output:
            output.append(value)
    return output

def min_max_normalization(values, min_range=0.0, max_range=1.0):
    v_min = min(values)
    v_max = max(values)
    output = [(min_range + (((v - v_min) * (max_range - min_range)) / (v_max - v_min))) for v in values] 
    return output


def z_score_normalization(values):
    mean_val = sum(values) / len(values)
    variance = sum((v - mean_val)**2 for v in values) / len(values)
    std_val = math.sqrt(variance)

    output =[(v - mean_val) / std_val for v in values] 
    return output
