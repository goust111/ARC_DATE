import numpy as np
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Move colored fragments into matching framed containers by relative position.

    Input structure:
    - black background (dominant color).
    - one or more hollow rectangular grey frames; each frame has a single marker pixel
      adjacent to its outer top-left side indicating which fragment color it accepts.
    - several small 4-connected same-color fragments (2..5 pixels) in colors matching
      marker colors.

    Rule:
    - copy the input grid, clear fragment cells and keep the grey frame borders and
      markers unchanged.
    - for each fragment, match its color to the frame marker color.
    - compute the fragment centroid versus the frame center: if |dr| >= |dc|, assign it to
      up/down by dr sign (up on tie or when dr < 0), else assign it to left/right by dc sign.
    - translate each group of same frame and same direction fragments inside the frame so the
      first fragment of the group touches the corresponding inner wall and later fragments stack
      toward the center with no gap, ordered by area descending.

    Key insight:
    - color only chooses the target frame, while relative direction to the frame center decides
      which inner wall and stack axis the fragment occupies.
    """

    def _components(grid, sel_mask, colors=None):
        H, W = grid.shape
        vis = np.zeros((H, W), dtype=bool)
        out = []
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

        for r in range(H):
            for c in range(W):
                if not sel_mask[r, c] or vis[r, c]:
                    continue
                if colors is not None and grid[r, c] not in colors:
                    continue
                color = grid[r, c]
                comp = []
                stack = [(r, c)]
                vis[r, c] = True
                idx = 0
                min_r, max_r = r, r
                min_c, max_c = c, c

                while idx < len(stack):
                    cr, cc = stack[idx]
                    idx += 1
                    comp.append((cr, cc))
                    if cr < min_r:
                        min_r = cr
                    if cr > max_r:
                        max_r = cr
                    if cc < min_c:
                        min_c = cc
                    if cc > max_c:
                        max_c = cc

                    for dr, dc in dirs:
                        nr, nc = cr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and (not vis[nr, nc]) and sel_mask[nr, nc] and grid[nr, nc] == color:
                            vis[nr, nc] = True
                            stack.append((nr, nc))

                out.append((color, np.array(comp, dtype=np.int64), (min_r, min_c, max_r, max_c)))
        return out

    def _bbox_from_cells(cells):
        rows = cells[:, 0]
        cols = cells[:, 1]
        min_r = int(rows.min())
        max_r = int(rows.max())
        min_c = int(cols.min())
        max_c = int(cols.max())
        return min_r, min_c, max_r, max_c, int(rows.size)

    def _find_frames(inp):
        H, W = inp.shape
        bg = int(np.bincount(inp.ravel()).argmax())
        mask = inp != bg
        comps = _components(inp, mask)

        frames = []
        for color, cells, (min_r, min_c, max_r, max_c) in comps:
            h = max_r - min_r + 1
            w = max_c - min_c + 1
            if h < 3 or w < 3:
                continue

            ok = True
            # perimeter check
            for c in range(min_c, max_c + 1):
                if inp[min_r, c] != color or inp[max_r, c] != color:
                    ok = False
                    break
            if not ok:
                continue
            for r in range(min_r + 1, max_r):
                if inp[r, min_c] != color or inp[r, max_c] != color:
                    ok = False
                    break
            if not ok:
                continue

            if h > 2 and w > 2:
                inner = inp[min_r + 1:max_r, min_c + 1:max_c]
                if np.any(inner != bg):
                    continue
            else:
                continue

            marker_pos = None
            candidates = [(min_r - 1, min_c), (min_r, min_c - 1)]
            for mr, mc in candidates:
                if 0 <= mr < H and 0 <= mc < W:
                    mv = inp[mr, mc]
                    if mv != bg and mv != color:
                        marker_pos = (mr, mc, int(mv))
                        break
            if marker_pos is None:
                continue

            frames.append({
                "color": int(color),
                "bbox": (int(min_r), int(min_c), int(max_r), int(max_c)),
                "marker": marker_pos,
            })

        return bg, frames

    def _classify_direction(rows, cols, frame):
        min_r, min_c, max_r, max_c = frame["bbox"]
        center_r = (min_r + max_r) * 0.5
        center_c = (min_c + max_c) * 0.5
        cr = float(rows.mean())
        cc = float(cols.mean())
        dr = cr - center_r
        dc = cc - center_c
        if abs(dr) >= abs(dc):
            return "up" if dr < 0 else "down"
        return "left" if dc < 0 else "right"

    bg, frames = _find_frames(grid)
    if not frames:
        return np.array(grid, copy=True)

    marker_to_frame = {f["marker"][2]: f for f in frames}
    frame_colors = {f["color"] for f in frames}
    H, W = grid.shape
    out = np.array(grid, copy=True)

    marker_cells = np.zeros((H, W), dtype=bool)
    for f in frames:
        mr, mc, _ = f["marker"]
        marker_cells[mr, mc] = True

    movable_mask = (grid != bg) & (~marker_cells)
    if frame_colors:
        for c in frame_colors:
            movable_mask &= (grid != c)

    comps = _components(grid, movable_mask)
    if not comps:
        return out

    fragments_by_frame_dir = {}
    for color, cells, _ in comps:
        if color not in marker_to_frame:
            # unknown color: keep unchanged
            continue
        frame = marker_to_frame[color]
        rows = cells[:, 0]
        cols = cells[:, 1]
        area = int(rows.size)
        if area == 0:
            continue
        direction = _classify_direction(rows, cols, frame)
        min_r, min_c, max_r, max_c, area = _bbox_from_cells(cells)
        h = max_r - min_r + 1
        w = max_c - min_c + 1

        # clear original piece position first
        out[rows, cols] = bg

        key = (frame["marker"][2], direction)
        fragments_by_frame_dir.setdefault(key, []).append(
            {
                "color": int(color),
                "cells": cells,
                "min_r": min_r,
                "min_c": min_c,
                "max_r": max_r,
                "max_c": max_c,
                "h": h,
                "w": w,
                "area": area,
                "rows": rows,
                "cols": cols,
            }
        )

    # place fragments wall-by-wall, stacked inward by descending area
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inner_top = r0 + 1
        inner_left = c0 + 1
        inner_bottom = r1 - 1
        inner_right = c1 - 1

        for direction, cursor_key in (
            ("up", "up"),
            ("down", "down"),
            ("left", "left"),
            ("right", "right"),
        ):
            pass

        stacks = {
            "up": 0,
            "down": 0,
            "left": 0,
            "right": 0,
        }

        for direction in ("up", "down", "left", "right"):
            group = fragments_by_frame_dir.get((frame["marker"][2], direction), [])
            if not group:
                continue
            group.sort(key=lambda item: item["area"], reverse=True)
            for piece in group:
                min_r = piece["min_r"]
                min_c = piece["min_c"]
                max_r = piece["max_r"]
                max_c = piece["max_c"]
                h = piece["h"]
                w = piece["w"]
                rows = piece["rows"]
                cols = piece["cols"]
                color = piece["color"]

                # axis translation from wall inward
                if direction == "up":
                    dy = (inner_top + stacks["up"]) - min_r
                    dx = 0
                    # keep horizontal position, clamp to frame interior if needed
                    if min_c + dx < inner_left:
                        dx += inner_left - (min_c + dx)
                    if max_c + dx > inner_right:
                        dx -= (max_c + dx) - inner_right
                    stacks["up"] += h
                elif direction == "down":
                    dy = (inner_bottom - stacks["down"]) - max_r
                    dx = 0
                    if min_c + dx < inner_left:
                        dx += inner_left - (min_c + dx)
                    if max_c + dx > inner_right:
                        dx -= (max_c + dx) - inner_right
                    stacks["down"] += h
                elif direction == "left":
                    dx = (inner_left + stacks["left"]) - min_c
                    dy = 0
                    if min_r + dy < inner_top:
                        dy += inner_top - (min_r + dy)
                    if max_r + dy > inner_bottom:
                        dy -= (max_r + dy) - inner_bottom
                    stacks["left"] += w
                else:  # right
                    dx = (inner_right - stacks["right"]) - max_c
                    dy = 0
                    if min_r + dy < inner_top:
                        dy += inner_top - (min_r + dy)
                    if max_r + dy > inner_bottom:
                        dy -= (max_r + dy) - inner_bottom
                    stacks["right"] += w

                rr = rows + dy
                cc = cols + dx

                # keep within interior if accidental overflow occurs
                if rr.size == 0:
                    continue
                rr = np.clip(rr, inner_top, inner_bottom)
                cc = np.clip(cc, inner_left, inner_right)

                out[rr, cc] = color

    return out


def generate_input() -> np.ndarray:
    rng = random.Random()

    # Palette: skip black/grey ids and stay in small ARC-like color space.
    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    bg = 0
    frame_color = 5

    H = rng.randint(18, 30)
    W = rng.randint(18, 30)
    grid = np.full((H, W), bg, dtype=int)

    occupied = np.zeros((H, W), dtype=bool)

    def can_place_mask(mask):
        if not mask:
            return True
        rr, cc = zip(*mask)
        return not occupied[rr, cc].any()

    def apply_cells(mask, value):
        if not mask:
            return
        rr, cc = zip(*mask)
        grid[list(rr), list(cc)] = value
        occupied[list(rr), list(cc)] = True

    # choose frames
    num_frames = rng.randint(1, 3)
    frames = []

    for i in range(num_frames):
        if len(palette) < num_frames:
            break

        tries = 0
        placed = False
        while tries < 200 and not placed:
            tries += 1
            fh = rng.randint(10, min(16, H - 4))
            fw = rng.randint(10, min(16, W - 4))
            if fh >= H or fw >= W:
                continue

            # keep at least one-cell margin around frame for markers/nearby fragments
            min_r = 2
            max_r = H - fh - 2
            min_c = 2
            max_c = W - fw - 2
            if max_r < min_r or max_c < min_c:
                continue

            r0 = rng.randint(min_r, max_r)
            c0 = rng.randint(min_c, max_c)
            r1 = r0 + fh - 1
            c1 = c0 + fw - 1
            cells = []
            for c in range(c0, c1 + 1):
                cells.append((r0, c))
                cells.append((r1, c))
            for r in range(r0 + 1, r1):
                cells.append((r, c0))
                cells.append((r, c1))

            if not can_place_mask(cells):
                continue

            # top-left marker placement (outside the box)
            marker_side = rng.choice(["up", "left"])
            if marker_side == "up" and r0 > 0:
                marker = (r0 - 1, c0)
            elif c0 > 0:
                marker = (r0, c0 - 1)
            elif r0 > 0:
                marker = (r0 - 1, c0)
            else:
                continue

            if occupied[marker]:
                continue

            marker_color = palette[i]
            if marker_color == frame_color:
                continue

            marker_r, marker_c = marker
            apply_cells([(r0, c) for c in range(c0, c1 + 1)], frame_color)
            apply_cells([(r1, c) for c in range(c0, c1 + 1)], frame_color)
            apply_cells([(r, c0) for r in range(r0 + 1, r1)], frame_color)
            apply_cells([(r, c1) for r in range(r0 + 1, r1)], frame_color)
            apply_cells([(marker_r, marker_c)], marker_color)

            # mark interior reserved for now as blocked
            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    occupied[r, c] = True

            # keep marker pixel outside the frame but still treated as occupied
            # keep a light halo outside for cleaner spacing
            for rr in (r0 - 1, r0, r0 + 1):
                for cc in (c0 - 1, c0, c0 + 1):
                    if 0 <= rr < H and 0 <= cc < W:
                        if not occupied[rr, cc]:
                            occupied[rr, cc] = False

            frames.append({
                "color": marker_color,
                "bbox": (r0, c0, r1, c1),
                "marker": (marker_r, marker_c),
            })
            placed = True

        if not placed:
            break

    if not frames:
        return grid

    # Unblock frame interiors for potential fragment placement regions but keep borders and markers blocked.
    for f in frames:
        r0, c0, r1, c1 = f["bbox"]
        occupied[r0 : r1 + 1, c0 : c1 + 1] = False
        # re-block borders and marker explicitly
        for c in range(c0, c1 + 1):
            occupied[r0, c] = True
            occupied[r1, c] = True
        for r in range(r0 + 1, r1):
            occupied[r, c0] = True
            occupied[r, c1] = True
        mr, mc = f["marker"]
        occupied[mr, mc] = True

    # connected-shape helper (4-neighbor, area 2..5, bounded footprint for easier stacking)
    def random_shape(area: int):
        for _ in range(80):
            cells = [(0, 0)]
            frontier = [(0, 0)]
            while len(cells) < area:
                base = rng.choice(frontier)
                br, bc = base
                options = [(br + 1, bc), (br - 1, bc), (br, bc + 1), (br, bc - 1)]
                options = [p for p in options if p not in cells]
                if not options:
                    break
                nr, nc = rng.choice(options)
                cells.append((nr, nc))
                frontier.append((nr, nc))

            arr = np.array(cells, dtype=int)
            h = arr[:, 0].max() - arr[:, 0].min() + 1
            w = arr[:, 1].max() - arr[:, 1].min() + 1
            if arr.shape[0] == area and h <= 3 and w <= 3:
                return cells
        return [(0, 0), (0, 1)]

    def direction_of_placement(frame, direction):
        r0, c0, r1, c1 = frame["bbox"]
        center_r = (r0 + r1) / 2.0
        center_c = (c0 + c1) / 2.0

        if direction == "up":
            return rng.randint(0, max(0, r0 - 1)), rng.randint(max(0, c0 - 3), min(W - 1, c1 + 3))
        if direction == "down":
            return rng.randint(r1 + 1, H - 1), rng.randint(max(0, c0 - 3), min(W - 1, c1 + 3))
        if direction == "left":
            return rng.randint(max(0, r0 - 3), min(H - 1, r1 + 3)), rng.randint(0, max(0, c0 - 1))
        return rng.randint(max(0, r0 - 3), min(H - 1, r1 + 3)), rng.randint(c1 + 1, W - 1)

    def centroid(rows, cols):
        return float(np.mean(rows)), float(np.mean(cols))

    def matches_direction(rows, cols, frame, direction):
        r0, c0, r1, c1 = frame["bbox"]
        center_r = (r0 + r1) * 0.5
        center_c = (c0 + c1) * 0.5
        dr = float(np.mean(rows)) - center_r
        dc = float(np.mean(cols)) - center_c
        if abs(dr) >= abs(dc):
            wanted = "up" if dr < 0 else "down"
        else:
            wanted = "left" if dc < 0 else "right"
        return wanted == direction

    def place_shape(mask_cells, anchor_r, anchor_c):
        return {(anchor_r + r, anchor_c + c) for r, c in mask_cells}

    dir_choices = ["up", "down", "left", "right"]

    total_fragments = 0
    for f in frames:
        # choose number of pieces for this frame
        n_frag = rng.randint(1, 4)
        # ensure each generated frame tends to expose multiple direction branches
        dirs = dir_choices.copy()
        rng.shuffle(dirs)
        planned_dirs = [dirs[i] for i in range(min(n_frag, len(dirs)))]
        while len(planned_dirs) < n_frag:
            planned_dirs.append(rng.choice(dir_choices))

        for d in planned_dirs:
            area = rng.randint(2, 5)
            rel_cells = random_shape(area)
            rel_arr = np.array(rel_cells, dtype=int)
            hr, hc = rel_arr[:, 0].min(), rel_arr[:, 1].min()
            rel_cells = [(r - hr, c - hc) for r, c in rel_cells]
            rel_arr = np.array(rel_cells, dtype=int)
            sh = rel_arr[:, 0].max() - rel_arr[:, 0].min() + 1
            sw = rel_arr[:, 1].max() - rel_arr[:, 1].min() + 1

            placed = False
            for _ in range(300):
                seed_r, seed_c = direction_of_placement(f, d)

                # randomize with global jitter around direction seed
                seed_r = int(np.clip(seed_r + rng.randint(-2, 2), 0, H - 1))
                seed_c = int(np.clip(seed_c + rng.randint(-2, 2), 0, W - 1))

                ar = seed_r - rel_arr[:, 0].mean()
                ac = seed_c - rel_arr[:, 1].mean()
                ar = int(np.floor(ar))
                ac = int(np.floor(ac))

                if ar < 0 or ac < 0 or ar + sh > H or ac + sw > W:
                    continue

                candidate = {(ar + r, ac + c) for r, c in rel_cells}
                if not candidate:
                    continue

                if any(occupied[rr, cc] for rr, cc in candidate):
                    continue

                rr = np.array(sorted({p[0] for p in candidate}), dtype=int)
                cc = np.array(sorted({p[1] for p in candidate}), dtype=int)
                if not matches_direction(rr, cc, f, d):
                    continue

                apply_cells(candidate, f["color"])
                total_fragments += 1
                placed = True
                # add a soft exclusion around the fragment
                for rr_i, cc_i in candidate:
                    occupied[rr_i, cc_i] = True
                break

            if not placed:
                # fallback: try any free anchor with matching direction
                for rr0 in range(H):
                    for cc0 in range(W):
                        ar = rr0
                        ac = cc0
                        if ar + sh > H or ac + sw > W:
                            continue
                        candidate = {(ar + r, ac + c) for r, c in rel_cells}
                        if any(occupied[rr2, cc2] for rr2, cc2 in candidate):
                            continue
                        rr = np.array([p[0] for p in candidate], dtype=int)
                        cc = np.array([p[1] for p in candidate], dtype=int)
                        if matches_direction(rr, cc, f, d):
                            apply_cells(candidate, f["color"])
                            for rr_i, cc_i in candidate:
                                occupied[rr_i, cc_i] = True
                            total_fragments += 1
                            placed = True
                            break
                    if placed:
                        break

    if total_fragments == 0:
        # fallback: ensure at least one valid fragment exists
        first = frames[0]
        d = "up"
        area = 2
        rel_cells = [(0, 0), (0, 1)]
        rel_arr = np.array(rel_cells)
        for _ in range(60):
            r0 = max(1, first["bbox"][0] - 3)
            c0 = max(0, first["bbox"][1] - 3)
            cand = {(r0 + r, c0 + c) for r, c in rel_cells}
            if any(occupied[rr, cc] for rr, cc in cand):
                continue
            r1, c1 = first["bbox"][0], first["bbox"][1]
            center_r = (r1 + first["bbox"][2]) / 2.0
            center_c = (c1 + first["bbox"][3]) / 2.0
            cr = (r0 + r0) / 2.0
            cc = (c0 + (c0 + 1)) / 2.0
            if ((abs(cr - center_r) >= abs(cc - center_c) and cr < center_r) or abs(cr - center_r) < abs(cc - center_c) and cc < center_c):
                apply_cells(cand, first["color"])
                break

    return grid


if __name__ == "__main__":
    # Optional local sanity check
    inp = generate_input()
    out = main(inp)
    assert inp.shape == out.shape
    print(inp.shape, "->", out.shape)
