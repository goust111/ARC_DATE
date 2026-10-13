import hashlib
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

_TRANSFORMS = [
    ("r0", lambda m: m.copy()),
    ("r1", lambda m: np.rot90(m, 1)),
    ("r2", lambda m: np.rot90(m, 2)),
    ("r3", lambda m: np.rot90(m, 3)),
    ("r0f", lambda m: np.fliplr(m)),
    ("r1f", lambda m: np.fliplr(np.rot90(m, 1))),
    ("r2f", lambda m: np.fliplr(np.rot90(m, 2))),
    ("r3f", lambda m: np.fliplr(np.rot90(m, 3))),
]

_SEEN_OUTPUT_SIGS: set[str] = set()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _crop_array(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _compress_runs(mask: np.ndarray) -> np.ndarray:
    mask = _crop_mask(mask)
    row_keep = [0]
    for r in range(1, mask.shape[0]):
        if not np.array_equal(mask[r], mask[r - 1]):
            row_keep.append(r)
    mask = mask[row_keep, :]

    col_keep = [0]
    for c in range(1, mask.shape[1]):
        if not np.array_equal(mask[:, c], mask[:, c - 1]):
            col_keep.append(c)
    return mask[:, col_keep]


def _named_variants(mask: np.ndarray):
    out = []
    seen = set()
    for name, fn in _TRANSFORMS:
        variant = fn(mask)
        key = _mask_key(variant)
        if key not in seen:
            seen.add(key)
            out.append((name, variant.copy()))
    return out


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _separator_spans(indices: list[int], length: int) -> list[tuple[int, int]]:
    indices = sorted(int(x) for x in indices)
    assert len(indices) == 2
    spans = []
    start = 0
    for idx in indices:
        assert start < idx
        spans.append((start, idx))
        start = idx + 1
    assert start < length
    spans.append((start, length))
    sizes = {end - start for start, end in spans}
    assert len(sizes) == 1
    return spans


def _discover_tiles(grid: np.ndarray):
    h, w = grid.shape
    sep_rows = [r for r in range(h) if np.all(grid[r, :] == GREY)]
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == GREY)]
    row_spans = _separator_spans(sep_rows, h)
    col_spans = _separator_spans(sep_cols, w)
    tile_h = row_spans[0][1] - row_spans[0][0]
    tile_w = col_spans[0][1] - col_spans[0][0]
    return row_spans, col_spans, tile_h, tile_w


def _extract_tile(grid: np.ndarray, pos: tuple[int, int], row_spans, col_spans) -> np.ndarray:
    r0, r1 = row_spans[pos[0]]
    c0, c1 = col_spans[pos[1]]
    return grid[r0:r1, c0:c1]


def _classify_tile(tile: np.ndarray) -> str:
    colors = set(int(v) for v in np.unique(tile) if int(v) != BLACK)
    if colors == {RED}:
        return "prototype"
    if colors == {BLUE, GREEN}:
        return "template"
    if colors == {YELLOW}:
        return "candidate"
    raise AssertionError(f"unexpected tile colors: {sorted(colors)}")


def _place_colored_mask(tile_size: int, mask: np.ndarray, color: int, *, avoid_top_left: bool) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = mask.shape
    max_r = tile_size - h
    max_c = tile_size - w
    assert max_r >= 0 and max_c >= 0
    offsets = [(r0, c0) for r0 in range(max_r + 1) for c0 in range(max_c + 1)]
    if avoid_top_left and len(offsets) > 1:
        offsets = [p for p in offsets if p != (0, 0)]
    r0, c0 = random.choice(offsets)
    tile = np.zeros((tile_size, tile_size), dtype=int)
    tile[r0 : r0 + h, c0 : c0 + w][mask] = int(color)
    return tile, (r0, c0)


def _place_sprite(tile_size: int, sprite: np.ndarray, *, avoid_top_left: bool) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = sprite.shape
    max_r = tile_size - h
    max_c = tile_size - w
    assert max_r >= 0 and max_c >= 0
    offsets = [(r0, c0) for r0 in range(max_r + 1) for c0 in range(max_c + 1)]
    if avoid_top_left and len(offsets) > 1:
        offsets = [p for p in offsets if p != (0, 0)]
    r0, c0 = random.choice(offsets)
    tile = np.zeros((tile_size, tile_size), dtype=int)
    patch = tile[r0 : r0 + h, c0 : c0 + w]
    patch[sprite != BLACK] = sprite[sprite != BLACK]
    return tile, (r0, c0)


def _tile_origin(tile_pos: tuple[int, int], tile_size: int) -> tuple[int, int]:
    return tile_pos[0] * (tile_size + 1), tile_pos[1] * (tile_size + 1)


def _grid_signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def main(grid: np.ndarray) -> np.ndarray:
    """Use the red prototype's run-compressed shape to decide which yellow tiles receive the separate blue-green stamp.

    Input structure:
    - A black background grid is divided into a 3x3 board of equal tiles by full grey separator rows and columns.
    - Exactly one tile contains a red prototype object, exactly one non-adjacent tile contains a blue-green template sprite, and the other seven tiles each contain one yellow candidate object.
    - Prototype and candidates are single-color 4-connected shapes on black; the template is a two-color sprite on black.

    Rule:
    - Crop the red prototype and every yellow candidate to their tight bounding boxes, then run-compress by merging adjacent identical rows and adjacent identical columns.
    - A candidate matches iff its compressed mask equals the prototype's compressed mask under some rotation and/or reflection.
    - For every matching candidate tile, erase the yellow object and paste the cropped template sprite in its original orientation with the template bbox aligned to the tile's top-left corner.
    - For every non-matching candidate tile, erase it to black; prototype tile, template tile, and grey separators stay unchanged.

    Key insight:
    - The red object is only a shape discriminator after run compression, while the blue-green object is an independent stamp that is copied into the tiles selected by that discriminator.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    row_spans, col_spans, tile_h, tile_w = _discover_tiles(g)
    assert tile_h == tile_w

    prototype_pos = None
    template_pos = None
    candidate_positions = []
    for tr in range(3):
        for tc in range(3):
            pos = (tr, tc)
            kind = _classify_tile(_extract_tile(g, pos, row_spans, col_spans))
            if kind == "prototype":
                assert prototype_pos is None
                prototype_pos = pos
            elif kind == "template":
                assert template_pos is None
                template_pos = pos
            else:
                candidate_positions.append(pos)

    assert prototype_pos is not None and template_pos is not None and len(candidate_positions) == 7

    prototype_tile = _extract_tile(g, prototype_pos, row_spans, col_spans)
    template_tile = _extract_tile(g, template_pos, row_spans, col_spans)

    prototype_mask = _compress_runs(prototype_tile == RED)
    prototype_sigs = {_mask_key(variant) for _name, variant in _named_variants(prototype_mask)}

    stamp = _crop_array(template_tile, template_tile != BLACK)
    sh, sw = stamp.shape
    assert sh <= tile_h and sw <= tile_w

    out = g.copy()
    for pos in candidate_positions:
        r0, r1 = row_spans[pos[0]]
        c0, c1 = col_spans[pos[1]]
        out[r0:r1, c0:c1] = BLACK
        candidate_tile = _extract_tile(g, pos, row_spans, col_spans)
        candidate_sig = _mask_key(_compress_runs(candidate_tile == YELLOW))
        if candidate_sig in prototype_sigs:
            out[r0 : r0 + sh, c0 : c0 + sw] = stamp

    return out


def _random_connected_mask(
    *,
    min_area: int,
    max_area: int,
    max_h: int,
    max_w: int,
    reject_full_rectangle: bool,
) -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(2, max_h)
        w = random.randint(2, max_w)
        if h * w < min_area:
            continue
        area = random.randint(min_area, min(max_area, h * w))
        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        cells = {seed}
        mask[seed] = True
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cell = random.choice(tuple(frontier))
            cells.add(cell)
            mask[cell] = True
        if len(cells) != area:
            continue
        mask = _crop_mask(mask)
        if min(mask.shape) < 2:
            continue
        if reject_full_rectangle and mask.all():
            continue
        if len(_components_mask(mask)) != 1:
            continue
        return mask
    return None


def _make_prototype_mask() -> np.ndarray | None:
    for _ in range(200):
        mask = _random_connected_mask(
            min_area=3,
            max_area=6,
            max_h=4,
            max_w=4,
            reject_full_rectangle=True,
        )
        if mask is None:
            continue
        variants = _named_variants(mask)
        if len(variants) < 2:
            continue
        return mask
    return None


def _stretch_mask(mask: np.ndarray, tile_size: int, *, force_extra: bool) -> tuple[np.ndarray, list[int], list[int]] | None:
    h, w = mask.shape
    for _ in range(200):
        row_factors = [random.choice([1, 2, 3]) for _ in range(h)]
        col_factors = [random.choice([1, 2, 3]) for _ in range(w)]
        if force_extra and max(row_factors + col_factors) == 1:
            if random.random() < 0.5:
                row_factors[random.randrange(h)] = 2
            else:
                col_factors[random.randrange(w)] = 2
        stretched = np.repeat(np.repeat(mask, row_factors, axis=0), col_factors, axis=1)
        if stretched.shape[0] > tile_size or stretched.shape[1] > tile_size:
            continue
        area = int(stretched.sum())
        if not (3 <= area <= 12):
            continue
        return stretched, row_factors, col_factors
    return None


def _make_template_sprite(tile_size: int) -> np.ndarray | None:
    for _ in range(200):
        mask = _random_connected_mask(
            min_area=4,
            max_area=8,
            max_h=min(4, tile_size),
            max_w=min(4, tile_size),
            reject_full_rectangle=False,
        )
        if mask is None:
            continue
        h, w = mask.shape
        if h == tile_size and w == tile_size:
            continue
        coords = np.argwhere(mask)
        green_count = random.randint(1, len(coords) - 1)
        green_indices = set(random.sample(range(len(coords)), green_count))
        sprite = np.zeros(mask.shape, dtype=int)
        for idx, (r, c) in enumerate(coords):
            sprite[r, c] = GREEN if idx in green_indices else BLUE
        return sprite
    return None


def _make_distractor(proto_sigs, tile_size: int, *, target_area: int | None) -> tuple[np.ndarray, dict] | None:
    for _ in range(300):
        base = _random_connected_mask(
            min_area=3,
            max_area=6,
            max_h=min(4, tile_size),
            max_w=min(4, tile_size),
            reject_full_rectangle=True,
        )
        if base is None:
            continue
        if target_area is not None and int(base.sum()) != int(target_area):
            continue
        variant_name, variant = random.choice(_named_variants(base))
        stretched = _stretch_mask(variant, tile_size, force_extra=random.random() < 0.5)
        if stretched is None:
            continue
        raw, row_factors, col_factors = stretched
        sig = _mask_key(_compress_runs(raw))
        if sig in proto_sigs:
            continue
        meta = {
            "match": False,
            "variant": variant_name,
            "row_factors": row_factors,
            "col_factors": col_factors,
        }
        return raw, meta
    return None


def _nonadjacent_positions():
    tiles = [(r, c) for r in range(3) for c in range(3)]
    pairs = []
    for a in tiles:
        for b in tiles:
            if a == b:
                continue
            if max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1:
                pairs.append((a, b))
    return pairs


_NONADJACENT_PAIRS = _nonadjacent_positions()


def _generate_instance():
    tile_size = random.randint(5, 7)
    grid_size = tile_size * 3 + 2
    grid = np.zeros((grid_size, grid_size), dtype=int)
    grid[tile_size, :] = GREY
    grid[2 * tile_size + 1, :] = GREY
    grid[:, tile_size] = GREY
    grid[:, 2 * tile_size + 1] = GREY

    prototype_pos, template_pos = random.choice(_NONADJACENT_PAIRS)
    all_positions = [(r, c) for r in range(3) for c in range(3)]
    candidate_positions = [p for p in all_positions if p not in {prototype_pos, template_pos}]
    random.shuffle(candidate_positions)

    prototype_mask = _make_prototype_mask()
    template_sprite = _make_template_sprite(tile_size)
    if prototype_mask is None or template_sprite is None:
        return None

    prototype_tile, prototype_offset = _place_colored_mask(
        tile_size, prototype_mask, RED, avoid_top_left=True
    )
    template_tile, template_offset = _place_sprite(
        tile_size, template_sprite, avoid_top_left=True
    )

    proto_comp = _compress_runs(prototype_mask)
    proto_sigs = {_mask_key(variant) for _name, variant in _named_variants(proto_comp)}

    variant_options = [(name, variant) for name, variant in _named_variants(prototype_mask) if name != "r0"]
    if not variant_options:
        return None

    match_count = random.randint(3, 5)
    matches_meta = {}

    special_name, special_variant = random.choice(variant_options)
    stretched = _stretch_mask(special_variant, tile_size, force_extra=True)
    if stretched is None:
        return None
    special_raw, special_rows, special_cols = stretched
    matches_meta[candidate_positions[0]] = {
        "raw": special_raw,
        "match": True,
        "variant": special_name,
        "row_factors": special_rows,
        "col_factors": special_cols,
        "requires_transform": True,
        "requires_compression": max(special_rows + special_cols) > 1,
    }

    for pos in candidate_positions[1:match_count]:
        variant_name, variant = random.choice(_named_variants(prototype_mask))
        stretched = _stretch_mask(variant, tile_size, force_extra=random.random() < 0.75)
        if stretched is None:
            return None
        raw, row_factors, col_factors = stretched
        matches_meta[pos] = {
            "raw": raw,
            "match": True,
            "variant": variant_name,
            "row_factors": row_factors,
            "col_factors": col_factors,
            "requires_transform": variant_name != "r0",
            "requires_compression": max(row_factors + col_factors) > 1,
        }

    distractors_meta = {}
    remaining = candidate_positions[match_count:]
    for idx, pos in enumerate(remaining):
        target_area = int(prototype_mask.sum()) if idx == 0 else None
        made = _make_distractor(proto_sigs, tile_size, target_area=target_area)
        if made is None:
            return None
        raw, meta = made
        meta["raw"] = raw
        distractors_meta[pos] = meta

    meta = {
        "tile_size": tile_size,
        "prototype_pos": prototype_pos,
        "template_pos": template_pos,
        "prototype_offset": prototype_offset,
        "template_offset": template_offset,
        "prototype_mask": prototype_mask,
        "template_sprite": template_sprite,
        "stamp": template_sprite.copy(),
        "candidates": {},
    }
    meta["candidates"].update(matches_meta)
    meta["candidates"].update(distractors_meta)

    for pos, tile in [(prototype_pos, prototype_tile), (template_pos, template_tile)]:
        r0, c0 = _tile_origin(pos, tile_size)
        grid[r0 : r0 + tile_size, c0 : c0 + tile_size] = tile

    for pos in candidate_positions:
        candidate_tile, offset = _place_colored_mask(
            tile_size, meta["candidates"][pos]["raw"], YELLOW, avoid_top_left=True
        )
        meta["candidates"][pos]["offset"] = offset
        r0, c0 = _tile_origin(pos, tile_size)
        grid[r0 : r0 + tile_size, c0 : c0 + tile_size] = candidate_tile

    return grid, meta


def _validate_instance(grid: np.ndarray, meta: dict) -> None:
    row_spans, col_spans, tile_h, tile_w = _discover_tiles(grid)
    assert tile_h == meta["tile_size"] and tile_w == meta["tile_size"]

    kinds = {}
    for tr in range(3):
        for tc in range(3):
            pos = (tr, tc)
            tile = _extract_tile(grid, pos, row_spans, col_spans)
            assert GREY not in tile
            kind = _classify_tile(tile)
            kinds[pos] = kind

    assert sum(kind == "prototype" for kind in kinds.values()) == 1
    assert sum(kind == "template" for kind in kinds.values()) == 1
    assert sum(kind == "candidate" for kind in kinds.values()) == 7
    assert kinds[meta["prototype_pos"]] == "prototype"
    assert kinds[meta["template_pos"]] == "template"
    assert max(
        abs(meta["prototype_pos"][0] - meta["template_pos"][0]),
        abs(meta["prototype_pos"][1] - meta["template_pos"][1]),
    ) > 1
    assert meta["template_offset"] != (0, 0)

    prototype_tile = _extract_tile(grid, meta["prototype_pos"], row_spans, col_spans)
    template_tile = _extract_tile(grid, meta["template_pos"], row_spans, col_spans)
    assert np.array_equal(_crop_mask(prototype_tile == RED), meta["prototype_mask"])
    assert np.array_equal(_crop_array(template_tile, template_tile != BLACK), meta["stamp"])

    proto_sigs = {
        _mask_key(variant) for _name, variant in _named_variants(_compress_runs(meta["prototype_mask"]))
    }
    seen_match = 0
    seen_nonmatch = 0
    saw_transform = False
    saw_compression = False
    for pos, cand_meta in meta["candidates"].items():
        tile = _extract_tile(grid, pos, row_spans, col_spans)
        raw = _crop_mask(tile == YELLOW)
        assert 3 <= int(raw.sum()) <= 12
        sig = _mask_key(_compress_runs(raw))
        actual_match = sig in proto_sigs
        assert actual_match == cand_meta["match"]
        if cand_meta["match"]:
            seen_match += 1
        else:
            seen_nonmatch += 1
        saw_transform |= bool(cand_meta.get("requires_transform"))
        saw_compression |= bool(cand_meta.get("requires_compression"))

    assert seen_match >= 1
    assert seen_nonmatch >= 2
    assert saw_transform
    assert saw_compression

    out = main(grid)
    assert np.array_equal(out[grid == GREY], grid[grid == GREY])
    assert np.array_equal(
        _extract_tile(out, meta["prototype_pos"], row_spans, col_spans),
        prototype_tile,
    )
    assert np.array_equal(
        _extract_tile(out, meta["template_pos"], row_spans, col_spans),
        template_tile,
    )

    stamp = meta["stamp"]
    sh, sw = stamp.shape
    for pos, cand_meta in meta["candidates"].items():
        tile = _extract_tile(out, pos, row_spans, col_spans)
        if cand_meta["match"]:
            expected = np.zeros((tile_h, tile_w), dtype=int)
            expected[:sh, :sw] = stamp
            assert np.array_equal(tile, expected)
        else:
            assert np.all(tile == BLACK)


def generate_input() -> np.ndarray:
    for _ in range(500):
        made = _generate_instance()
        if made is None:
            continue
        grid, meta = made
        _validate_instance(grid, meta)
        output_sig = _grid_signature(main(grid))
        if output_sig in _SEEN_OUTPUT_SIGS:
            continue
        _SEEN_OUTPUT_SIGS.add(output_sig)
        return grid
    raise RuntimeError("failed to generate a valid non-duplicate instance")
