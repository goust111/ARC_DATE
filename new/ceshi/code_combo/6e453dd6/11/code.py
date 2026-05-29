# Combo variant of 6e453dd6 (index 11)
# combo_spec: D3/3+D7/1
# Name: 限长渐变射线

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


for _parent in Path(__file__).resolve().parents:
    _arc_utils = _parent / "solvers" / "arc_utils.py"
    if _arc_utils.exists():
        sys.path.append(str(_arc_utils.parent))
        break

from arc_utils import components_mask, paste


# Design analysis for combo D3/3 + D7/1
# - Concrete trace: after sliding, suppose one sprite becomes a 5x4 black shell whose interior hole is 3x2
#   and touches the column just inside the separator. In the original task, the three hole rows would each
#   paint a solid red sweep all the way to the right border. In this combo, the hole area is 6, so each of
#   those rows paints exactly six cells, colored [red, red, orange, orange, yellow, yellow]. For a 2x2 hole,
#   the same rows paint four cells [red, red, orange, yellow]. The end point and the color progression are
#   therefore coupled: the gradient stops exactly where the area-encoded beam stops.
# - Complexity calibration: the generator limits emitter holes to a tiny catalog {(3x1), (2x2), (3x2)},
#   giving beam lengths {3, 4, 6}. That creates visible variation without introducing too many count values.
#   The right pane is widened to 8-9 cells, so every beam leaves a background tail and cannot collapse into
#   the original "fill to border" behavior.
# - Implicit-assumption audit for the original main():
#   * Independent horizontal sliding still works because the generator keeps sprites in disjoint row bands.
#   * Hole detection as "background components not touching the left-pane boundary" still works because
#     emitters remain closed shells and distractors remain open or solid shapes.
#   * The old assignment `out[r, sep + 1 :] = red` is invalid twice: D7 shortens the beam, and D3 replaces
#     a constant color with a position-dependent sequence. The solver must compute a full per-hole sequence.
#   * Row collisions can only be tolerated if they imply the exact same gradient sequence, not merely the
#     same length. The solver therefore stores the whole color tuple per emitting row.
# - Placement legality: generation is output-first. Sprites are stamped flush against the separator in
#   separated row bands, then each input sprite is unslid left by a bounded shift. Every emitter area is at
#   most 6, strictly below the chosen right-pane width, so no gradient beam overflows or degenerates.
# - Visual observability and cleanliness: orange and yellow appear only on the right-side beams, never inside
#   the left-pane sprites. Because beams stay within the rows of their source holes and always leave a visible
#   background suffix, a human can compare hole size against beam reach and also see the red->orange->yellow
#   ordering directly from only a few examples.
# - New solver-side assumptions introduced by the combo:
#   * Every emitting hole area must fit within the right pane; the generator enforces this with the fixed
#     emitter catalog and wider right pane.
#   * Beam color depends only on distance from the separator, not on row index or on any hidden numeric order.
#   * If two emitting holes ever share a row, they must encode the same gradient sequence; the generator avoids
#     such conflicts by giving sprites non-overlapping row bands.


BG = 6
BLACK = 0
RED = 2
YELLOW = 4
SEP_COLOR = 5
ORANGE = 7

GRADIENT = (RED, ORANGE, YELLOW)

H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 15, 20
SEP_MIN, SEP_MAX = 6, 10
RIGHT_W_MIN, RIGHT_W_MAX = 8, 9

N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
MIN_BAND_GAP = 1

EXTRA_EMITTERS_PROB = 0.3
FLUSH_PROB = 0.35

EMITTER_SPECS = (
    (3, 1),  # area 3
    (2, 2),  # area 4
    (3, 2),  # area 6
)

MAX_RETRIES = 500


def _components4(mask):
    return components_mask(mask, connectivity=4)


def _find_separator_col(g: np.ndarray, sep_color: int = SEP_COLOR) -> int:
    _, w = g.shape
    cols = [c for c in range(w) if np.all(g[:, c] == sep_color)]
    assert len(cols) == 1, f"expected a unique full separator column, got {cols}"
    return int(cols[0])


def _background_color(g: np.ndarray, sep_col: int, sep_color: int = SEP_COLOR) -> int:
    h, w = g.shape
    border = []
    for c in range(w):
        if c != sep_col and g[0, c] != sep_color:
            border.append(int(g[0, c]))
        if c != sep_col and g[h - 1, c] != sep_color:
            border.append(int(g[h - 1, c]))
    for r in range(h):
        if g[r, 0] != sep_color:
            border.append(int(g[r, 0]))
        if g[r, w - 1] != sep_color:
            border.append(int(g[r, w - 1]))
    assert border, "expected non-separator border pixels to infer the background"
    vals, counts = np.unique(np.array(border, dtype=int), return_counts=True)
    winners = vals[counts == counts.max()]
    assert len(winners) == 1, f"background color must be unique on the border, got {winners.tolist()}"
    return int(winners[0])


def _gradient_sequence(ray_len: int) -> tuple[int, ...]:
    assert ray_len >= 1, "beam length must be positive"
    return tuple(GRADIENT[min(2, (3 * idx) // ray_len)] for idx in range(ray_len))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black sprites to the grey divider, then fire area-limited beams with a red-to-yellow gradient.

    Input structure:
    - A single full-height grey separator splits the board into a left sprite pane and an empty right pane.
    - Black connected components occupy the left pane on one background color; the right pane begins as pure background.

    Rule:
    - Shift every black connected component horizontally until its rightmost black pixel sits against the separator.
    - In that slid layout, detect enclosed background holes and keep only the holes that touch the inner column next to the separator.
    - For each row of such a hole, paint a beam starting immediately right of the separator; its length equals the hole's area, and its color changes with distance as red, then orange, then yellow.

    Key insight: The emitting rows still come from separator-adjacent enclosed holes, but now the hole contributes two linked signals at once: its area sets the stopping point, and that same span is visually marked by a left-to-right color gradient.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    sep = _find_separator_col(g, sep_color=SEP_COLOR)
    bg = _background_color(g, sep_col=sep, sep_color=SEP_COLOR)

    out = g.copy()

    left = out[:, :sep].copy()
    comps = _components4(left == BLACK)
    left[left == BLACK] = bg

    target_right = sep - 1
    for comp in comps:
        max_c = max(c for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        for r, c in comp:
            nc = c + delta
            assert 0 <= nc < sep, f"shifted pixel {(r, c)} left the pane"
            left[r, nc] = BLACK

    out[:, :sep] = left

    trigger_col = sep - 2
    beam_by_row: dict[int, tuple[int, ...]] = {}
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        if not any(c == trigger_col for _, c in pts):
            continue
        hole_rows = sorted({int(r) for r, _ in pts})
        ray = _gradient_sequence(len(pts))
        for r in hole_rows:
            prev = beam_by_row.get(r)
            assert prev in (None, ray), f"conflicting beam sequences on row {r}: {prev} vs {ray}"
            beam_by_row[r] = ray

    for r, ray in sorted(beam_by_row.items()):
        end = sep + 1 + len(ray)
        assert end <= w, f"beam on row {r} exceeds right pane: end={end}, width={w}"
        assert np.all(out[r, sep + 1 : end] == bg), f"beam path on row {r} must be background before emission"
        out[r, sep + 1 : end] = np.array(ray, dtype=int)

    return out


def _scatter(total: int, slots: int) -> list[int]:
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _min_emitters_for(n: int) -> int:
    return (3 * n + 4) // 5


def _min_rows_needed(n_emitters: int, n_total: int) -> int:
    min_emitter_h = min(hole_h + 2 for hole_h, _ in EMITTER_SPECS)
    min_distractor_h = 2
    return n_emitters * min_emitter_h + (n_total - n_emitters) * min_distractor_h + (n_total + 1) * MIN_BAND_GAP


def _choose_grid_dims() -> tuple[int, int, int, int]:
    h = random.randint(H_MIN, H_MAX)
    right_w = random.randint(RIGHT_W_MIN, RIGHT_W_MAX)
    valid_seps = [s for s in range(SEP_MIN, SEP_MAX + 1) if W_MIN <= s + 1 + right_w <= W_MAX]
    sep = random.choice(valid_seps)
    w = sep + 1 + right_w
    return h, w, sep, right_w


def _choose_emitter_flags(h: int) -> list[bool]:
    valid_counts = [
        n
        for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1)
        if _min_rows_needed(_min_emitters_for(n), n) <= h
    ]
    n = random.choice(valid_counts)

    min_em = _min_emitters_for(n)
    max_em = min_em
    for k in range(min_em + 1, n + 1):
        if _min_rows_needed(k, n) <= h:
            max_em = k
    if max_em > min_em and random.random() < EXTRA_EMITTERS_PROB:
        n_emitters = random.randint(min_em + 1, max_em)
    else:
        n_emitters = min_em

    flags = [True] * n_emitters + [False] * (n - n_emitters)
    random.shuffle(flags)
    return flags


def _make_emitter_patch(max_sw: int) -> tuple[np.ndarray, dict[str, object]]:
    hole_h, hole_w = random.choice(EMITTER_SPECS)
    sh, sw = hole_h + 2, hole_w + 2
    assert sw <= max_sw, f"emitter width {sw} exceeds left-pane allowance {max_sw}"

    patch = np.full((sh, sw), BG, dtype=int)
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[:, 0] = BLACK
    patch[:, -1] = BLACK

    area = hole_h * hole_w
    return patch, {
        "kind": "emitter",
        "hole_rows": tuple(range(1, sh - 1)),
        "area": area,
        "hole_dims": (hole_h, hole_w),
    }


def _make_distractor_patch(max_sw: int) -> tuple[np.ndarray, dict[str, object]]:
    kind = random.choice(["l_shape", "c_left", "c_right", "solid"])

    if kind == "solid":
        sh = random.randint(2, 3)
        sw = random.randint(2, min(3, max_sw))
        patch = np.full((sh, sw), BLACK, dtype=int)
        return patch, {"kind": kind}

    if kind == "l_shape":
        sh = random.randint(2, 5)
        sw = random.randint(2, min(4, max_sw))
        patch = np.full((sh, sw), BG, dtype=int)
        patch[:, -1] = BLACK
        if random.random() < 0.5:
            patch[0, :] = BLACK
        else:
            patch[-1, :] = BLACK
        return patch, {"kind": kind}

    sh = random.randint(3, 5)
    sw = random.randint(3, min(4, max_sw))
    patch = np.full((sh, sw), BG, dtype=int)
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    if kind == "c_left":
        patch[:, -1] = BLACK
    else:
        patch[:, 0] = BLACK
    return patch, {"kind": kind}


def _layout_rows(h: int, sprite_heights: list[int]) -> list[int] | None:
    n = len(sprite_heights)
    min_total = sum(sprite_heights) + (n + 1) * MIN_BAND_GAP
    if min_total > h:
        return None

    gaps = [MIN_BAND_GAP] * (n + 1)
    extra = h - min_total
    for i, delta in enumerate(_scatter(extra, n + 1)):
        gaps[i] += delta

    starts = []
    r = gaps[0]
    for i, sh in enumerate(sprite_heights):
        starts.append(r)
        r += sh + gaps[i + 1]
    return starts


def _build_sprites(sep: int, flags: list[bool]) -> list[tuple[np.ndarray, dict[str, object]]] | None:
    sprites = []
    max_sw = sep - 1
    for is_emitter in flags:
        patch, meta = _make_emitter_patch(max_sw) if is_emitter else _make_distractor_patch(max_sw)
        if patch.shape[1] > max_sw:
            return None
        sprites.append((patch, meta))
    return sprites


def _render_output(
    h: int,
    w: int,
    sep: int,
    right_w: int,
    row_starts: list[int],
    sprites: list[tuple[np.ndarray, dict[str, object]]],
) -> tuple[np.ndarray, list[tuple[np.ndarray, int, int]], list[tuple[int, int]]]:
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    placed: list[tuple[np.ndarray, int, int]] = []
    emitter_signatures: list[tuple[int, int]] = []
    for r0, (patch, meta) in zip(row_starts, sprites):
        sh, sw = patch.shape
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        if meta["kind"] == "emitter":
            ray = _gradient_sequence(int(meta["area"]))
            assert len(ray) < right_w, f"beam length {len(ray)} must stop before right border width {right_w}"
            for lr in meta["hole_rows"]:
                out[r0 + int(lr), sep + 1 : sep + 1 + len(ray)] = np.array(ray, dtype=int)
            hole_h, _ = meta["hole_dims"]
            emitter_signatures.append((int(hole_h), int(meta["area"])))
        placed.append((patch, r0, c0))
    return out, placed, emitter_signatures


def _has_area_disambiguation(emitter_signatures: list[tuple[int, int]]) -> bool:
    by_height: dict[int, set[int]] = {}
    for hole_h, area in emitter_signatures:
        by_height.setdefault(hole_h, set()).add(area)
    return any(len(areas) >= 2 for areas in by_height.values())


def _unslide_sprites(h: int, w: int, sep: int, sprites: list[tuple[np.ndarray, int, int]]) -> np.ndarray | None:
    n = len(sprites)
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR

    flush_i = random.randrange(n) if random.random() < FLUSH_PROB else -1
    for i, (patch, r0, c0) in enumerate(sprites):
        if c0 < 1:
            return None
        delta = 0 if i == flush_i else random.randint(1, c0)
        paste(inp, patch, r0, c0 - delta, transparent=BG)
    return inp


def _gen_attempt() -> np.ndarray | None:
    h, w, sep, right_w = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    sprites = _build_sprites(sep, flags)
    if sprites is None:
        return None

    row_starts = _layout_rows(h, [patch.shape[0] for patch, _ in sprites])
    if row_starts is None:
        return None

    out, placed, emitter_signatures = _render_output(h, w, sep, right_w, row_starts, sprites)
    if not emitter_signatures or not _has_area_disambiguation(emitter_signatures):
        return None

    inp = _unslide_sprites(h, w, sep, placed)
    if inp is None:
        return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6e453dd6 combo input")
