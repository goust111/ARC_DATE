from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG = 0
OCCLUDER = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
RETRY_LIMIT = 400


@dataclass(frozen=True)
class BandModel:
    flipped: bool
    offsets: tuple[int, ...]
    row_start: int
    row_end: int
    sequence: tuple[int, ...]

    @property
    def period(self) -> int:
        return len(self.sequence)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _flip_bbox(box: tuple[int, int, int, int], width: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return int(r0), int(width - 1 - c1), int(r1), int(width - 1 - c0)


def _strict_period(seq: tuple[int, ...]) -> int:
    n = len(seq)
    for p in range(1, n + 1):
        if n % p != 0:
            continue
        ok = True
        for i, color in enumerate(seq):
            if color != seq[i % p]:
                ok = False
                break
        if ok:
            return p
    return n


def _row_range(height: int, width: int, offsets: tuple[int, ...]) -> tuple[int, int]:
    row_start = max(0, -max(offsets))
    row_end = min(height - 1, width - 1 - min(offsets))
    return int(row_start), int(row_end)


def _render_canonical(
    shape: tuple[int, int],
    offsets: tuple[int, ...],
    row_start: int,
    row_end: int,
    sequence: tuple[int, ...],
) -> np.ndarray:
    height, width = map(int, shape)
    out = np.zeros((height, width), dtype=int)
    period = len(sequence)
    for row in range(row_start, row_end + 1):
        color = int(sequence[(row - row_start) % period])
        for offset in offsets:
            col = row + offset
            if 0 <= col < width:
                out[row, col] = color
    return out


def _find_occluder_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey = grid == OCCLUDER
    box = _bbox(grey)
    assert box is not None, "expected one grey occluder"
    r0, c0, r1, c1 = box
    assert np.all(grey[r0 : r1 + 1, c0 : c1 + 1]), "occluder must be a solid rectangle"
    assert int(grey.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1), "grey must appear only in the occluder"
    assert (r1 - r0 + 1) in (3, 4) and (c1 - c0 + 1) in (3, 4), "occluder must be 3x3 or 4x4"
    return box


def _candidate_models(grid: np.ndarray) -> list[BandModel]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = map(int, g.shape)
    occluder = _find_occluder_bbox(g)

    models: list[BandModel] = []
    seen: set[BandModel] = set()
    for flipped in (False, True):
        canonical = np.fliplr(g) if flipped else g
        canonical_box = _flip_bbox(occluder, width) if flipped else occluder

        visible = (canonical != BG) & (canonical != OCCLUDER)
        coords = np.argwhere(visible)
        if coords.size == 0:
            continue

        offsets = sorted({int(c - r) for r, c in coords})
        if not (1 <= len(offsets) <= 2):
            continue
        if offsets[-1] - offsets[0] != len(offsets) - 1:
            continue

        anti_span = int((coords[:, 0] + coords[:, 1]).max() - (coords[:, 0] + coords[:, 1]).min())
        if offsets[-1] - offsets[0] >= anti_span:
            continue

        offset_tuple = tuple(offsets)
        row_start, row_end = _row_range(height, width, offset_tuple)
        if row_start > row_end:
            continue

        expected_band = np.zeros_like(visible, dtype=bool)
        for row in range(row_start, row_end + 1):
            for offset in offset_tuple:
                col = row + offset
                if 0 <= col < width:
                    expected_band[row, col] = True
        if np.any(visible & ~expected_band):
            continue

        row_colors: dict[int, int] = {}
        ok = True
        for row in range(row_start, row_end + 1):
            colors = {
                int(canonical[row, row + offset])
                for offset in offset_tuple
                if 0 <= row + offset < width and int(canonical[row, row + offset]) not in (BG, OCCLUDER)
            }
            if len(colors) > 1:
                ok = False
                break
            if colors:
                row_colors[row] = colors.pop()
        if not ok or len(row_colors) < 2:
            continue

        period_limit = min(4, row_end - row_start + 1)
        sequence: tuple[int, ...] | None = None
        for period in range(1, period_limit + 1):
            phase_colors: dict[int, int] = {}
            consistent = True
            for row, color in row_colors.items():
                phase = (row - row_start) % period
                prior = phase_colors.get(phase)
                if prior is not None and prior != color:
                    consistent = False
                    break
                phase_colors[phase] = color
            if not consistent or len(phase_colors) != period:
                continue
            candidate_sequence = tuple(phase_colors[i] for i in range(period))
            if _strict_period(candidate_sequence) != period:
                continue
            sequence = candidate_sequence
            break
        if sequence is None:
            continue

        rendered = _render_canonical(g.shape, offset_tuple, row_start, row_end, sequence)
        if not np.array_equal(rendered[canonical != OCCLUDER], canonical[canonical != OCCLUDER]):
            continue

        r0, c0, r1, c1 = canonical_box
        occluded_rows = [
            row
            for row in range(row_start, row_end + 1)
            if any(r0 <= row <= r1 and c0 <= row + offset <= c1 for offset in offset_tuple)
        ]
        if not occluded_rows:
            continue
        if row_start >= min(occluded_rows) or max(occluded_rows) >= row_end:
            continue
        if not all(
            int(canonical[row, row + offset]) == OCCLUDER
            for row in occluded_rows
            for offset in offset_tuple
            if 0 <= row + offset < width and r0 <= row <= r1 and c0 <= row + offset <= c1
        ):
            continue

        model = BandModel(
            flipped=flipped,
            offsets=offset_tuple,
            row_start=row_start,
            row_end=row_end,
            sequence=sequence,
        )
        if model not in seen:
            seen.add(model)
            models.append(model)
    return models


def _solve(grid: np.ndarray) -> np.ndarray:
    models = _candidate_models(grid)
    assert len(models) == 1, f"expected a unique band model, found {len(models)}"
    model = models[0]

    clean = _render_canonical(grid.shape, model.offsets, model.row_start, model.row_end, model.sequence)
    return np.fliplr(clean) if model.flipped else clean


def main(grid: np.ndarray) -> np.ndarray:
    """Restore the diagonal periodic color band hidden behind the grey rectangle.

    Input structure
    - A mostly black grid contains a narrow color band running along one diagonal direction.
    - The band has width 1 or 2 cells and uses a repeating row-periodic color schedule.
    - A single solid grey 3x3 or 4x4 rectangle covers one middle segment of the band.

    Rule
    - Locate the grey rectangle and treat it as the only occluded region.
    - Infer the band direction, occupied diagonal offsets, and the smallest repeating color period from the visible band cells outside the rectangle.
    - Repaint the hidden band cells inside the rectangle from that periodic rule, and turn every occluder cell not belonging to the band back to black.

    Key insight
    - The hidden content is recovered by periodic extrapolation along the diagonal trajectory, not by a geometric symmetry copy.
    """
    return _solve(np.asarray(grid, dtype=int))


def _sequence_with_exact_period(period: int, distinct_count: int) -> tuple[int, ...]:
    choices = random.sample(PALETTE, distinct_count)
    for _ in range(200):
        seq = tuple(random.choice(choices) for _ in range(period))
        if len(set(seq)) != distinct_count:
            continue
        if _strict_period(seq) != period:
            continue
        return seq
    raise RuntimeError("failed to sample a non-degenerate periodic color sequence")


def _occluder_candidates(
    clean: np.ndarray,
    offsets: tuple[int, ...],
    row_start: int,
    row_end: int,
    period: int,
    size: int,
) -> list[tuple[int, int, int, int]]:
    height, width = map(int, clean.shape)
    candidates: list[tuple[int, int, int, int]] = []
    for r0 in range(height - size + 1):
        for c0 in range(width - size + 1):
            r1 = r0 + size - 1
            c1 = c0 + size - 1
            occluded_rows = [
                row
                for row in range(row_start, row_end + 1)
                if any(r0 <= row <= r1 and c0 <= row + offset <= c1 for offset in offsets)
            ]
            if not occluded_rows:
                continue
            first = min(occluded_rows)
            last = max(occluded_rows)
            if first - row_start < period or row_end - last < period:
                continue

            band_pixels = 0
            for row in occluded_rows:
                for offset in offsets:
                    col = row + offset
                    if 0 <= col < width and r0 <= row <= r1 and c0 <= col <= c1:
                        band_pixels += 1
            if band_pixels < size:
                continue
            if band_pixels >= size * size:
                continue
            candidates.append((r0, c0, r1, c1))
    return candidates


def _visible_offsets_after_occlusion(
    shape: tuple[int, int],
    offsets: tuple[int, ...],
    box: tuple[int, int, int, int],
) -> set[int]:
    height, width = map(int, shape)
    r0, c0, r1, c1 = box
    visible: set[int] = set()
    row_start, row_end = _row_range(height, width, offsets)
    for row in range(row_start, row_end + 1):
        for offset in offsets:
            col = row + offset
            if 0 <= col < width and not (r0 <= row <= r1 and c0 <= col <= c1):
                visible.add(offset)
    return visible


def _sample_design() -> tuple[np.ndarray, np.ndarray]:
    height = random.randint(13, 18)
    width = random.randint(13, 18)
    flipped = random.choice([False, True])
    band_width = random.choice([1, 2])
    period = random.choice([2, 3, 4])
    distinct_count = random.choice([2, 3] if period >= 3 else [2])
    sequence = _sequence_with_exact_period(period, distinct_count)
    occluder_size = random.choice([3, 4])

    offset_options = []
    for start_offset in range(-3, 4):
        offsets = tuple(start_offset + i for i in range(band_width))
        row_start, row_end = _row_range(height, width, offsets)
        if row_end - row_start + 1 >= 2 * period + occluder_size:
            offset_options.append(offsets)
    if not offset_options:
        raise RuntimeError("no feasible band offsets for sampled size")

    offsets = random.choice(offset_options)
    row_start, row_end = _row_range(height, width, offsets)
    clean_canonical = _render_canonical((height, width), offsets, row_start, row_end, sequence)

    occluder_options = _occluder_candidates(
        clean_canonical,
        offsets,
        row_start,
        row_end,
        period,
        occluder_size,
    )
    occluder_options = [
        box
        for box in occluder_options
        if _visible_offsets_after_occlusion((height, width), offsets, box) == set(offsets)
    ]
    if not occluder_options:
        raise RuntimeError("no feasible occluder placement for sampled design")

    box = random.choice(occluder_options)
    r0, c0, r1, c1 = box
    inp_canonical = clean_canonical.copy()
    inp_canonical[r0 : r1 + 1, c0 : c1 + 1] = OCCLUDER

    if flipped:
        return np.fliplr(inp_canonical), np.fliplr(clean_canonical)
    return inp_canonical, clean_canonical


def generate_input() -> np.ndarray:
    for _ in range(RETRY_LIMIT):
        try:
            inp, expected = _sample_design()
        except RuntimeError:
            continue

        try:
            actual = main(inp)
        except Exception:
            continue

        if not np.array_equal(actual, expected):
            continue
        if len(_candidate_models(inp)) != 1:
            continue

        grey_box = _find_occluder_bbox(inp)
        r0, c0, r1, c1 = grey_box
        grey_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if grey_area not in (9, 16):
            continue
        if len(np.unique(inp)) < 4:
            continue

        return inp
    raise RuntimeError("failed to generate a valid periodic-band occlusion input")
