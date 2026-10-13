from __future__ import annotations

import random

import numpy as np


BG = 0
SIDES = ("left", "right", "top", "bottom")
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 300


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color == bg:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "touches_edge": bool(
                        (arr[:, 0] == 0).any()
                        or (arr[:, 0] == h - 1).any()
                        or (arr[:, 1] == 0).any()
                        or (arr[:, 1] == w - 1).any()
                    ),
                }
            )

    return comps


def _ring_mask(layers: int, side: str) -> np.ndarray:
    size = 2 * int(layers) + 1
    mid = size // 2
    mask = np.zeros((size, size), dtype=bool)

    for offset in range(layers):
        r0 = offset
        c0 = offset
        r1 = size - 1 - offset
        c1 = size - 1 - offset
        mask[r0, c0 : c1 + 1] = True
        mask[r1, c0 : c1 + 1] = True
        mask[r0 : r1 + 1, c0] = True
        mask[r0 : r1 + 1, c1] = True

        if side == "left":
            mask[mid, c0] = False
        elif side == "right":
            mask[mid, c1] = False
        elif side == "top":
            mask[r0, mid] = False
        else:
            mask[r1, mid] = False

    return mask


def _ring_patch(color: int, layers: int, side: str) -> np.ndarray:
    mask = _ring_mask(layers, side)
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    return patch


def _component_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = comp["coords"] - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _parse_channel(comp: dict, shape: tuple[int, int]) -> dict | None:
    h, w = shape
    r0, c0, r1, c1 = comp["bbox"]
    area = int(comp["coords"].shape[0])

    if r0 == r1 and area == c1 - c0 + 1:
        if c0 == 0 and c1 < w - 1:
            return {"side": "left", "color": int(comp["color"]), "row": int(r0), "inner": int(c1)}
        if c1 == w - 1 and c0 > 0:
            return {"side": "right", "color": int(comp["color"]), "row": int(r0), "inner": int(c0)}

    if c0 == c1 and area == r1 - r0 + 1:
        if r0 == 0 and r1 < h - 1:
            return {"side": "top", "color": int(comp["color"]), "col": int(c0), "inner": int(r1)}
        if r1 == h - 1 and r0 > 0:
            return {"side": "bottom", "color": int(comp["color"]), "col": int(c0), "inner": int(r0)}

    return None


def _parse_ring(comp: dict) -> dict | None:
    if bool(comp["touches_edge"]):
        return None

    r0, c0, r1, c1 = comp["bbox"]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height != width or height not in (3, 5):
        return None

    layers = (height - 1) // 2
    mask = _component_mask(comp)
    matches = [side for side in SIDES if np.array_equal(mask, _ring_mask(layers, side))]
    if len(matches) != 1:
        return None

    return {
        "side": matches[0],
        "color": int(comp["color"]),
        "layers": int(layers),
        "size": int(height),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "coords": comp["coords"],
    }


def _destination(channel: dict, size: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    radius = size // 2
    side = channel["side"]

    if side == "left":
        r0 = int(channel["row"]) - radius
        c0 = 0
    elif side == "right":
        r0 = int(channel["row"]) - radius
        c0 = w - size
    elif side == "top":
        r0 = 0
        c0 = int(channel["col"]) - radius
    else:
        r0 = h - size
        c0 = int(channel["col"]) - radius

    assert 0 <= r0 <= h - size and 0 <= c0 <= w - size, "ring destination must fit"
    return int(r0), int(c0)


def _draw_marker_extension(grid: np.ndarray, channel: dict, layers: int, color: int) -> None:
    h, w = grid.shape
    side = channel["side"]

    if side == "left":
        row = int(channel["row"])
        start = int(channel["inner"]) + 1
        assert start + layers - 1 < w
        grid[row, start : start + layers] = int(color)
        return

    if side == "right":
        row = int(channel["row"])
        end = int(channel["inner"]) - 1
        assert end - layers + 1 >= 0
        grid[row, end - layers + 1 : end + 1] = int(color)
        return

    if side == "top":
        col = int(channel["col"])
        start = int(channel["inner"]) + 1
        assert start + layers - 1 < h
        grid[start : start + layers, col] = int(color)
        return

    col = int(channel["col"])
    end = int(channel["inner"]) - 1
    assert end - layers + 1 >= 0
    grid[end - layers + 1 : end + 1, col] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each gapped ring onto its matching edge channel and append a layer-count marker at that channel's far end.

    Input structure:
    - The background is black.
    - Several monochrome one-pixel-wide straight channels touch exactly one grid edge;
      horizontal channels come from the left or right edge, and vertical channels
      come from the top or bottom edge.
    - Away from the edges, each remaining foreground component is a monochrome
      3x3 or 5x5 ring with a black center and exactly one missing side-middle
      pixel; 5x5 rings are nested double rings whose two layers share the same gap.

    Rule:
    - Use the ring's gap direction to choose the channel side: left-gap rings go
      to left-edge channels, right-gap rings to right-edge channels, top-gap rings
      to top-edge channels, and bottom-gap rings to bottom-edge channels.
    - Within that side, match each ring to the unique channel of the same color.
    - Remove the ring from its original location and repaint it flush against the
      corresponding edge, centered on the matched channel while preserving its
      shape and gap orientation.
    - At the channel's inward endpoint, extend the channel by 1 pixel for a single
      ring or 2 pixels for a double ring, using the same color.

    Key insight:
    The gap orientation identifies both the destination side and the movement
    direction, while the ring's visual depth (single vs double) determines how
    long the added endpoint marker must be.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(_dominant_color(g))
    comps = _find_components(g, bg)

    channels: dict[tuple[str, int], dict] = {}
    rings: list[dict] = []

    for comp in comps:
        channel = _parse_channel(comp, g.shape)
        if channel is not None:
            key = (str(channel["side"]), int(channel["color"]))
            assert key not in channels, "expected unique channel per side/color pair"
            channels[key] = channel
            continue

        ring = _parse_ring(comp)
        assert ring is not None, "unexpected foreground component"
        rings.append(ring)

    assert rings, "expected at least one ring"
    out = g.copy()

    for ring in rings:
        out[ring["coords"][:, 0], ring["coords"][:, 1]] = bg

    for ring in rings:
        key = (str(ring["side"]), int(ring["color"]))
        assert key in channels, "every ring needs a matching channel"
        channel = channels[key]
        size = int(ring["size"])
        r0, c0 = _destination(channel, size, g.shape)
        patch = _ring_patch(int(ring["color"]), int(ring["layers"]), str(ring["side"]))

        out[r0 : r0 + size, c0 : c0 + size] = bg
        mask = patch != BG
        out[r0 : r0 + size, c0 : c0 + size][mask] = int(ring["color"])
        _draw_marker_extension(out, channel, int(ring["layers"]), int(ring["color"]))

    return out


def _draw_channel(grid: np.ndarray, side: str, lane: int, start: int, end: int, color: int) -> None:
    if side == "left":
        grid[int(lane), int(start) : int(end) + 1] = int(color)
    elif side == "right":
        grid[int(lane), int(start) : int(end) + 1] = int(color)
    elif side == "top":
        grid[int(start) : int(end) + 1, int(lane)] = int(color)
    else:
        grid[int(start) : int(end) + 1, int(lane)] = int(color)


def _paint_patch(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    mask = patch != BG
    grid[int(r0) : int(r0) + patch.shape[0], int(c0) : int(c0) + patch.shape[1]][mask] = patch[mask]


def _can_place(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, padding: int = 1) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False

    for dr, dc in np.argwhere(patch != BG):
        rr = int(r0 + dr)
        cc = int(c0 + dc)
        r1 = max(0, rr - padding)
        r2 = min(h, rr + padding + 1)
        c1 = max(0, cc - padding)
        c2 = min(w, cc + padding + 1)
        if np.any(grid[r1:r2, c1:c2] != BG):
            return False

    return True


def _pick_spaced(candidates: list[int], count: int, min_gap: int) -> list[int] | None:
    if len(candidates) < count:
        return None

    for _ in range(200):
        picked = sorted(random.sample(candidates, count))
        if all(b - a >= min_gap for a, b in zip(picked, picked[1:])):
            return picked

    return None


def _balanced_sides(axis: str, count: int) -> list[str]:
    if axis == "horizontal":
        first, second = "left", "right"
    else:
        first, second = "top", "bottom"

    first_count = random.randint(1, count - 1)
    sides = [first] * first_count + [second] * (count - first_count)
    random.shuffle(sides)
    return sides


def _balanced_layers(count: int) -> list[int]:
    layers = [1, 2]
    while len(layers) < count:
        layers.append(random.choice([1, 2]))
    random.shuffle(layers)
    return layers


def _place_input_rings(inp: np.ndarray, specs: list[dict], axis: str, corridor: tuple[int, int]) -> bool:
    if axis == "horizontal":
        c0, c1 = corridor
        row_range = range(1, inp.shape[0] - 1)
        col_range = range(c0, c1 + 1)
    else:
        r0, r1 = corridor
        row_range = range(r0, r1 + 1)
        col_range = range(1, inp.shape[1] - 1)

    order = sorted(range(len(specs)), key=lambda i: -int(specs[i]["size"]))
    for idx in order:
        spec = specs[idx]
        patch = _ring_patch(int(spec["color"]), int(spec["layers"]), str(spec["side"]))
        size = int(spec["size"])
        candidates: list[tuple[int, int]] = []
        for r0 in row_range:
            if r0 + size > inp.shape[0] - 1:
                continue
            for c0 in col_range:
                if c0 + size > inp.shape[1] - 1:
                    continue
                candidates.append((int(r0), int(c0)))
        random.shuffle(candidates)

        placed = False
        for r0, c0 in candidates:
            if _can_place(inp, patch, r0, c0, padding=1):
                _paint_patch(inp, patch, r0, c0)
                spec["input_pos"] = (int(r0), int(c0))
                placed = True
                break

        if not placed:
            return False

    return True


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    axis = random.choice(["horizontal", "vertical"])
    count = random.randint(2, 4)
    colors = random.sample(list(range(1, 10)), count)
    layers = _balanced_layers(count)
    sizes = [2 * layer + 1 for layer in layers]
    sides = _balanced_sides(axis, count)

    if axis == "horizontal":
        h = random.randint(max(16, 5 * count), 29)
        w = random.randint(23, 29)
        left_reserved = random.randint(8, 10)
        right_reserved = random.randint(8, 10)
        if w - left_reserved - right_reserved < 8:
            return None

        corridor = (int(left_reserved), int(w - right_reserved - 1))
        lanes = _pick_spaced(list(range(2, h - 2)), count, 5)
        if lanes is None:
            return None

        specs: list[dict] = []
        for lane, color, side, layer, size in zip(lanes, colors, sides, layers, sizes):
            if side == "left":
                low = size - 1
                high = corridor[0] - 2 - layer
                if low > high:
                    return None
                inner = random.randint(low, high)
                specs.append(
                    {
                        "side": side,
                        "color": int(color),
                        "layers": int(layer),
                        "size": int(size),
                        "lane": int(lane),
                        "input_span": (0, int(inner)),
                        "output_span": (0, int(inner + layer)),
                    }
                )
            else:
                low = corridor[1] + 2 + layer
                high = w - size
                if low > high:
                    return None
                inner = random.randint(low, high)
                specs.append(
                    {
                        "side": side,
                        "color": int(color),
                        "layers": int(layer),
                        "size": int(size),
                        "lane": int(lane),
                        "input_span": (int(inner), w - 1),
                        "output_span": (int(inner - layer), w - 1),
                    }
                )

        inp = np.full((h, w), BG, dtype=int)
        out = np.full((h, w), BG, dtype=int)

        for spec in specs:
            start_in, end_in = spec["input_span"]
            start_out, end_out = spec["output_span"]
            _draw_channel(inp, str(spec["side"]), int(spec["lane"]), start_in, end_in, int(spec["color"]))
            _draw_channel(out, str(spec["side"]), int(spec["lane"]), start_out, end_out, int(spec["color"]))

        if not _place_input_rings(inp, specs, axis, corridor):
            return None

        for spec in specs:
            channel = {"side": str(spec["side"]), "color": int(spec["color"]), "row": int(spec["lane"]), "inner": int(spec["input_span"][1] if spec["side"] == "left" else spec["input_span"][0])}
            size = int(spec["size"])
            r0, c0 = _destination(channel, size, out.shape)
            patch = _ring_patch(int(spec["color"]), int(spec["layers"]), str(spec["side"]))
            out[r0 : r0 + size, c0 : c0 + size] = BG
            _paint_patch(out, patch, r0, c0)

        return inp, out

    h = random.randint(23, 29)
    w = random.randint(max(16, 5 * count), 29)
    top_reserved = random.randint(8, 10)
    bottom_reserved = random.randint(8, 10)
    if h - top_reserved - bottom_reserved < 8:
        return None

    corridor = (int(top_reserved), int(h - bottom_reserved - 1))
    lanes = _pick_spaced(list(range(2, w - 2)), count, 5)
    if lanes is None:
        return None

    specs = []
    for lane, color, side, layer, size in zip(lanes, colors, sides, layers, sizes):
        if side == "top":
            low = size - 1
            high = corridor[0] - 2 - layer
            if low > high:
                return None
            inner = random.randint(low, high)
            specs.append(
                {
                    "side": side,
                    "color": int(color),
                    "layers": int(layer),
                    "size": int(size),
                    "lane": int(lane),
                    "input_span": (0, int(inner)),
                    "output_span": (0, int(inner + layer)),
                }
            )
        else:
            low = corridor[1] + 2 + layer
            high = h - size
            if low > high:
                return None
            inner = random.randint(low, high)
            specs.append(
                {
                    "side": side,
                    "color": int(color),
                    "layers": int(layer),
                    "size": int(size),
                    "lane": int(lane),
                    "input_span": (int(inner), h - 1),
                    "output_span": (int(inner - layer), h - 1),
                }
            )

    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    for spec in specs:
        start_in, end_in = spec["input_span"]
        start_out, end_out = spec["output_span"]
        _draw_channel(inp, str(spec["side"]), int(spec["lane"]), start_in, end_in, int(spec["color"]))
        _draw_channel(out, str(spec["side"]), int(spec["lane"]), start_out, end_out, int(spec["color"]))

    if not _place_input_rings(inp, specs, axis, corridor):
        return None

    for spec in specs:
        channel = {"side": str(spec["side"]), "color": int(spec["color"]), "col": int(spec["lane"]), "inner": int(spec["input_span"][1] if spec["side"] == "top" else spec["input_span"][0])}
        size = int(spec["size"])
        r0, c0 = _destination(channel, size, out.shape)
        patch = _ring_patch(int(spec["color"]), int(spec["layers"]), str(spec["side"]))
        out[r0 : r0 + size, c0 : c0 + size] = BG
        _paint_patch(out, patch, r0, c0)

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue

        inp, expected = built
        got = main(inp)
        if np.array_equal(got, expected):
            return inp

    raise RuntimeError("failed to generate a valid puzzle instance")
