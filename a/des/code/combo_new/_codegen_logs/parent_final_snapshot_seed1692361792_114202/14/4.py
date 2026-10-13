import numpy as np


BLACK = 0
GREY = 5
PIECE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
GEN_RETRIES = 400


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                if rr > 0 and mask[rr - 1, cc] and not seen[rr - 1, cc]:
                    seen[rr - 1, cc] = True
                    stack.append((rr - 1, cc))
                if rr + 1 < height and mask[rr + 1, cc] and not seen[rr + 1, cc]:
                    seen[rr + 1, cc] = True
                    stack.append((rr + 1, cc))
                if cc > 0 and mask[rr, cc - 1] and not seen[rr, cc - 1]:
                    seen[rr, cc - 1] = True
                    stack.append((rr, cc - 1))
                if cc + 1 < width and mask[rr, cc + 1] and not seen[rr, cc + 1]:
                    seen[rr, cc + 1] = True
                    stack.append((rr, cc + 1))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _coords_to_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _mask_signature(mask: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    arr = np.asarray(mask, dtype=bool)
    return (int(arr.shape[0]), int(arr.shape[1]), tuple(int(v) for v in arr.ravel()))


def _flip_for_side(mask: np.ndarray, side: str) -> np.ndarray:
    arr = np.asarray(mask, dtype=bool)
    if side in ("top", "bottom"):
        return np.flipud(arr)
    return np.fliplr(arr)


def _primary_side(coords: np.ndarray, height: int, width: int) -> str | None:
    rows = coords[:, 0]
    cols = coords[:, 1]
    counts = {
        "top": int(np.sum(rows == 0)),
        "bottom": int(np.sum(rows == height - 1)),
        "left": int(np.sum(cols == 0)),
        "right": int(np.sum(cols == width - 1)),
    }
    side = max(counts, key=counts.get)
    best = counts[side]
    if best <= 0 or sum(value == best for value in counts.values()) != 1:
        return None
    return side


def _extract_frames(grid: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for comp in _components(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 5 or width < 5:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = _border_mask(height, width)
        if np.any((sub == GREY) & ~border):
            continue
        if np.any(sub[1:-1, 1:-1] != BLACK):
            continue
        missing = border & (sub != GREY)
        gaps_raw = _components(missing)
        if not (1 <= len(gaps_raw) <= 2):
            continue
        gaps = []
        ok = True
        for gap_coords in gaps_raw:
            side = _primary_side(gap_coords, height, width)
            if side is None:
                ok = False
                break
            abs_coords = gap_coords + np.array([r0, c0], dtype=int)
            gaps.append(
                {
                    "side": side,
                    "mask": _coords_to_mask(gap_coords),
                    "coords": abs_coords,
                }
            )
        if ok:
            frames.append({"bbox": (r0, c0, r1, c1), "gaps": gaps})
    return frames


def _extract_pieces(grid: np.ndarray) -> list[dict]:
    pieces = []
    mask = (grid != BLACK) & (grid != GREY)
    for comp in _components(mask):
        colors = np.unique(grid[comp[:, 0], comp[:, 1]])
        if len(colors) != 1:
            continue
        pieces.append(
            {
                "color": int(colors[0]),
                "coords": comp,
                "mask": _coords_to_mask(comp),
            }
        )
    return pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fit scattered colored border fragments back into the gaps of grey frames.

    Input structure:
    - The background is black.
    - Several grey hollow rectangles appear with 1-2 connected gaps cut out of their one-pixel border.
    - The inside of every frame is entirely black.
    - Outside the frames, each non-grey color forms one connected fragment whose shape matches exactly one gap after a flip.

    Rule:
    - Detect every grey rectangular frame and extract each missing border segment as a gap shape.
    - Determine the gap's main side from the frame edge it occupies: top or bottom gaps use a vertical flip, left or right gaps use a horizontal flip.
    - Match each colored fragment to the unique gap whose side-specific flip restores the gap shape.
    - Clear the fragment from its original location and paint its color into the corresponding black gap cells.

    Key insight:
    - The fragment shape identifies which gap it belongs to, but the gap's border side determines which flip axis aligns the fragment with the template.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _extract_frames(g)
    pieces = _extract_pieces(g)
    gaps = [gap for frame in frames for gap in frame["gaps"]]

    assert gaps, "expected at least one frame gap"
    assert len(gaps) == len(pieces), "number of fragments must equal number of gaps"

    unused = set(range(len(pieces)))
    placements = []
    for gap in gaps:
        matches = []
        for idx in sorted(unused):
            piece = pieces[idx]
            transformed = _flip_for_side(piece["mask"], gap["side"])
            if transformed.shape == gap["mask"].shape and np.array_equal(transformed, gap["mask"]):
                matches.append(idx)
        assert len(matches) == 1, "every gap must match exactly one fragment"
        idx = matches[0]
        unused.remove(idx)
        placements.append((gap, pieces[idx]))

    assert not unused, "every fragment must be used exactly once"

    for gap, piece in placements:
        out[piece["coords"][:, 0], piece["coords"][:, 1]] = BLACK
        out[gap["coords"][:, 0], gap["coords"][:, 1]] = int(piece["color"])

    return out


def _make_gap_mask(height: int, width: int, side: str, rng: np.random.Generator) -> np.ndarray | None:
    if side in ("top", "bottom"):
        max_run = min(5, width - 1)
    else:
        max_run = min(5, height - 1)
    if max_run < 3:
        return None

    for _ in range(80):
        run_len = int(rng.integers(3, max_run + 1))
        max_hook = min(2, run_len - 2)
        if max_hook < 1:
            continue
        hook_len = int(rng.integers(1, max_hook + 1))
        mask = np.zeros((height, width), dtype=bool)

        if side == "top":
            anchor_left = bool(rng.integers(0, 2))
            if anchor_left:
                mask[0, :run_len] = True
                mask[: hook_len + 1, 0] = True
            else:
                mask[0, width - run_len :] = True
                mask[: hook_len + 1, width - 1] = True
        elif side == "bottom":
            anchor_left = bool(rng.integers(0, 2))
            if anchor_left:
                mask[height - 1, :run_len] = True
                mask[height - hook_len - 1 :, 0] = True
            else:
                mask[height - 1, width - run_len :] = True
                mask[height - hook_len - 1 :, width - 1] = True
        elif side == "left":
            anchor_top = bool(rng.integers(0, 2))
            if anchor_top:
                mask[:run_len, 0] = True
                mask[0, : hook_len + 1] = True
            else:
                mask[height - run_len :, 0] = True
                mask[height - 1, : hook_len + 1] = True
        else:
            anchor_top = bool(rng.integers(0, 2))
            if anchor_top:
                mask[:run_len, width - 1] = True
                mask[0, width - hook_len - 1 :] = True
            else:
                mask[height - run_len :, width - 1] = True
                mask[height - 1, width - hook_len - 1 :] = True

        rows, cols = np.where(mask)
        counts = {
            "top": int(np.sum(rows == 0)),
            "bottom": int(np.sum(rows == height - 1)),
            "left": int(np.sum(cols == 0)),
            "right": int(np.sum(cols == width - 1)),
        }
        if _primary_side(np.argwhere(mask), height, width) == side:
            return mask
    return None


def _make_frame_spec(
    height: int,
    width: int,
    sides: list[str],
    rng: np.random.Generator,
    used_signatures: set[tuple[int, int, tuple[int, ...]]],
) -> dict | None:
    for _ in range(120):
        gap_union = np.zeros((height, width), dtype=bool)
        local_updates: set[tuple[int, int, tuple[int, ...]]] = set()
        gaps = []
        ok = True
        for side in sides:
            gap_full = _make_gap_mask(height, width, side, rng)
            if gap_full is None or np.any(gap_union & gap_full):
                ok = False
                break
            gap_crop = _coords_to_mask(np.argwhere(gap_full))
            sig = _mask_signature(gap_crop)
            rot_sig = _mask_signature(np.rot90(gap_crop, 2))
            if sig in used_signatures or rot_sig in used_signatures or sig in local_updates or rot_sig in local_updates:
                ok = False
                break
            gap_union |= gap_full
            local_updates.update([sig, rot_sig])
            gaps.append(
                {
                    "side": side,
                    "piece_mask": _flip_for_side(gap_crop, side),
                }
            )
        if not ok:
            continue
        used_signatures.update(local_updates)
        return {
            "height": height,
            "width": width,
            "border": _border_mask(height, width) & ~gap_union,
            "gaps": gaps,
        }
    return None


def _rect_is_free(occ: np.ndarray, r0: int, c0: int, height: int, width: int, pad: int) -> bool:
    h, w = occ.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h, r0 + height + pad)
    cc1 = min(w, c0 + width + pad)
    return not np.any(occ[rr0:rr1, cc0:cc1])


def _draw_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    rows, cols = np.where(mask)
    grid[r0 + rows, c0 + cols] = int(color)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    height = int(rng.integers(24, 31))
    width = int(rng.integers(24, 31))
    grid = np.full((height, width), BLACK, dtype=int)
    occ = np.zeros((height, width), dtype=bool)

    num_frames = int(rng.integers(2, 4))
    frame_specs = []
    used_signatures: set[tuple[int, int, tuple[int, ...]]] = set()

    for _ in range(num_frames):
        fh = int(rng.integers(6, 11))
        fw = int(rng.integers(6, 11))
        gap_count = int(rng.integers(1, 3))
        if gap_count == 2:
            sides = ["top", "bottom"] if bool(rng.integers(0, 2)) else ["left", "right"]
        else:
            sides = [SIDES[int(rng.integers(0, 4))]]
        spec = _make_frame_spec(fh, fw, sides, rng, used_signatures)
        if spec is None:
            return None
        frame_specs.append(spec)

    total_gaps = sum(len(spec["gaps"]) for spec in frame_specs)
    if total_gaps > len(PIECE_COLORS):
        return None

    for spec in sorted(frame_specs, key=lambda item: item["height"] * item["width"], reverse=True):
        placed = False
        for _ in range(200):
            r0 = int(rng.integers(0, height - spec["height"] + 1))
            c0 = int(rng.integers(0, width - spec["width"] + 1))
            if not _rect_is_free(occ, r0, c0, spec["height"], spec["width"], pad=1):
                continue
            spec["top"] = r0
            spec["left"] = c0
            sub = grid[r0 : r0 + spec["height"], c0 : c0 + spec["width"]]
            sub[spec["border"]] = GREY
            occ[r0 : r0 + spec["height"], c0 : c0 + spec["width"]] = True
            placed = True
            break
        if not placed:
            return None

    colors = list(rng.permutation(np.array(PIECE_COLORS, dtype=int))[:total_gaps])
    pieces = []
    color_idx = 0
    for spec in frame_specs:
        for gap in spec["gaps"]:
            pieces.append(
                {
                    "color": int(colors[color_idx]),
                    "mask": gap["piece_mask"],
                }
            )
            color_idx += 1

    pieces.sort(key=lambda item: item["mask"].shape[0] * item["mask"].shape[1], reverse=True)
    for piece in pieces:
        ph, pw = piece["mask"].shape
        placed = False
        for _ in range(250):
            r0 = int(rng.integers(0, height - ph + 1))
            c0 = int(rng.integers(0, width - pw + 1))
            if not _rect_is_free(occ, r0, c0, ph, pw, pad=1):
                continue
            _draw_mask(grid, piece["mask"], r0, c0, piece["color"])
            occ[r0 : r0 + ph, c0 : c0 + pw] = True
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    if np.any((grid == GREY) & (out != GREY)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
