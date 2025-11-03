def cluster_to_first_ps(new_y, new_x, existing_coords, radius=2):
    for y, x in existing_coords:
        if (new_y - y) ** 2 + (new_x - x) ** 2 <= radius ** 2:
            return y, x

    existing_coords.append((new_y, new_x))
    return new_y, new_x
