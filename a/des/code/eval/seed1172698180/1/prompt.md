你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="b10624e5" lines="24-39">
<description>
Replicate a motif across quadrants with scaling and reflection keyed to red anchor blocks.

Input structure:
- Grid divided into four quadrants by a full-row and full-column divider of a single non-bg color.
- Each quadrant contains a red (2) square anchor block of varying size.
- The top-left quadrant also has a colored motif attached to its red anchor.

Rule:
- Use the top-left quadrant's motif as the source pattern.
- For each other quadrant, scale the motif by the ratio of that quadrant's red block size to the source red block size.
- Reflect horizontally for right-side quadrants, vertically for bottom quadrants (both for bottom-right).
- Align the scaled/reflected motif so its red region overlaps the target anchor, then stamp non-bg pixels.

Key insight: Red block size encodes the scale factor; quadrant position encodes the reflection axes.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="eval" puzzle_id="1ae2feb7" lines="30-45">
<description>
Extend row bars periodically across the opposite side of a separator column.

Input structure:
- bg = background. A near-solid vertical separator divides the grid into two sides.
- One side contains horizontal colored bars; the other side is blank target space.
- Bars may be stacked by row and may sit at different distances from the separator.

Rule:
- Identify the separator as the column whose non-bg cells are one color.
- Treat the denser side as the bar side and collect each contiguous row run.
- Process bars from nearest to farthest from the separator; for each bar, paint its
  color every bar-length cells into the blank side without overwriting earlier paint.

Key insight: Each bar's length is the period for its extension, and nearer bars claim target cells first.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="36a08778" lines="8-21">
<description>
Simulate water falling from top-edge sources, pooling on shelves, and cascading off edges.

Input structure:
- bg = most common color; water = rarest non-bg; shelf = other non-bg.
- Sources: water clusters touching top row. Shelves: horizontal segments.

Rule:
- Water falls down from sources, painting bg cells as water.
- On hitting a shelf, it spreads across the shelf span plus one cell past each end.
- From those overflow positions, water resumes falling independently.

Key insight: Each shelf splits a stream into two at its edges, creating a branching waterfall tree.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="4c3d4a41" lines="15-28">
<description>
Fit a grey bar into a grey frame by aligning projections with colored columns, then stack colors above.

Input structure:
- bg, a grey structural color forming two components: a large rectangular frame (slot) and a smaller bar with upward projections.
- Colored pixels inside the slot mark target columns.

Rule:
- Move the bar upward into the slot so its bottom edge sits one cell above the slot's interior bottom.
- Shift it horizontally so the bar's upward projections align column-wise with the colored pixels in the slot.
- Stack the colored pixels immediately above the bar projections in each column.

Key insight: The bar's projections act as keys that must align with the colored-column "locks" in the slot.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="eval" puzzle_id="67e490f4" lines="8-22">
<description>
Fill holes inside a large template frame using shape-matched outside sprites.

Input structure:
- bg = most common color. One large monochrome frame object (template).
- Inside the frame: holes of various shapes (non-frame-color regions).
- Outside the frame: small monochrome sprites scattered on bg.

Rule:
- Crop to the frame's bounding box; fill entirely with frame color.
- For each hole, find an outside sprite whose shape matches (rotation/flip allowed).
- Fill the hole with the matched sprite's color. If multiple sprites match, use the most frequent color.

Key insight: Sprites act as shape-coded color keys — their shape identifies which hole they fill, their color determines the fill color.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="eval" puzzle_id="65b59efc" lines="9-24">
<description>
Decode a stamp map: read templates, markers, and palette from a tiled grid, then render stamps.

Input structure:
- Grid divided into K tile columns (each N wide) separated by 1-col dashed separators.
- Top band (rows 0..N-1): K template masks, one per tile column.
- Middle band (rows N+1..2N): marker pixels whose color picks a template.
- Bottom band (rows 2N+2..): K palette swatches giving the output color per template.
- Separator rows at rows N and 2N+1 (dashed pattern of sep color).

Rule:
- Each marker's local position (lr, lc) within its tile column determines which N×N output block to stamp.
- The marker's color identifies which template to use; the palette recolors the template.

Key insight: Marker color encodes template index; local position within tile column encodes stamp block.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="eval" puzzle_id="e376de54" lines="41-55">
<description>
Align parallel colored line segments so they all share the same start and end positions as a reference line.

Input structure:
- bg = most common color.
- Several parallel line segments (diagonal, horizontal, or vertical), each a different color, running in the same direction but starting/ending at different positions.

Rule:
- Detect the dominant direction shared by all segments.
- Pick the median segment (by perpendicular offset) as the reference.
- Shift every segment along the shared direction so its start aligns with the reference's start, then trim or extend it to match the reference's length.
- Grid size stays the same; isolated single pixels are untouched.

Key insight: The reference is the median line, not the longest; all segments are snapped to its extent regardless of their original length.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="3e6067c3" lines="7-20">
<description>
Connect framed squares via colored bridges following a control-pixel sequence.

Input structure:
- bg = background. Border color forms equal-sized framed squares with unique inner fill colors.
- Control row below squares: colored pixels read left-to-right encode visitation order.

Rule:
- Start at the square matching first control color c0.
- For each consecutive pair (ci, ci+1), find the ci+1-colored square aligned in the same row or column band, preferring nearest.
- Draw a bridge colored ci between the pair, spanning the gap with inner-square thickness.

Key insight: Row/column-band alignment plus proximity resolves ambiguity when inner colors repeat.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="eval" puzzle_id="58490d8a" lines="9-21">
<description>
Count colored objects and record tallies inside a legend box.

Input structure:
- A black rectangular legend box with key-color pixels in its leftmost non-black column.
- Outside the box, distinct objects of each key color are scattered across the grid.

Rule:
- For each key color, count separate same-colored objects outside the legend box.
- Output the legend box repainted: each key row gets one dot per object, spaced every other cell rightward from the key pixel.

Key insight: The legend is both the color index and the output canvas — object counts become dot sequences on the matching row.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="6ffbe589" lines="83-98">
<description>
Crop a board cluster and rotate its color layers based on outside cue sprites.

Input structure:
- bg = most common color.
- A main multi-color board cluster (largest foreground region).
- Small monochrome cue sprites outside the board for the colors that rotate.

Rule:
- Tight-crop around the board cluster.
- For each color layer inside the crop, find its same-color cue outside.
- Rotate that layer clockwise by (cue pixel count mod 4) × 90°.
- No outside cue → no rotation.

Key insight: Cue sprite size mod 4 encodes rotation amount for its matching color layer.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="291dc1e1" lines="7-22">
<description>
Reorder and stack rectangles by proximity to two colored edge lines.

Input structure:
- bg = most common color.
- Two edge lines (h + v) meet at a corner marked by a unique color.
- "Blue" line is color 1 if present, else the longer line; other is "red."
- Solid rectangles in the interior.

Rule:
- Sort rectangles by (distance to blue edge, distance to red edge).
- If the blue edge is vertical, rotate each rectangle 90° CCW.
- Stack top-to-bottom, centered horizontally.

Key insight: Edge-line colors encode a coordinate system; proximity to each axis determines stacking order and rotation.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="3a25b0d8" lines="9-22">
<description>
Fill holes in a target object with colors from a source object's subsprites, then tight-crop.

Input structure:
- bg = background. Two similar 8-connected objects share an outline color.
- Source has colored subsprites in cavities; target has empty cavities.

Rule:
- Find the two 8-connected objects.
- Target = object with more hole area. Align subsprites to holes (rotation/flip if needed).
- Fill each hole with its matched subsprite color; tight-crop.

Key insight: Hole centers in the target are matched to source subsprite centers under the best rotation/flip transform.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="eval" puzzle_id="eee78d87" lines="40-55">
<description>
Decode the colored 3x3 contour motif into a 16x16 periodic lattice with the motif stamped in maroon at the center.

Input structure:
- The grid is 6x6 with orange background and one non-orange color.
- The non-orange pixels have a tight 3x3 bbox containing one contour glyph.
- The glyph is one of the task's period-3 contour forms: plus, H, X, or four corners.

Rule:
- Decode the 3x3 glyph into the period-3 orange phase tile.
- Fill a black 16x16 canvas by repeating that orange phase tile.
- Tile the original 3x3 glyph 2x2 and recolor its true pixels to maroon in the centered 6x6 region.

Key insight:
- The colored object is not copied at its input location; it is a small contour code whose phase tells where orange appears in the repeated lattice.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="da515329" lines="11-24">
<description>
Grow four clockwise spiral wall paths from the tips of a teal cross.

Input structure:
- bg = 0 (black). A single teal (8) cross centered in the grid.

Rule:
- Each arm tip launches a "turtle" heading 90° right of its outward direction.
- Movement priority: turn right > go straight > turn left (left only if not on border).
- Normal step = 2 cells; step right after a turn = 1 cell.
- Each new cell must have a 1-cell orthogonal gap from all existing walls except the predecessor.

Key insight: The right-turn preference plus alternating 2/1 step lengths creates self-avoiding spirals that tile the grid with uniform spacing.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="fc7cae8d" lines="9-21">
<description>
Align a central rectangle to two colored frame-edge controls via rotation/reflection.

Input structure:
- bg grid with two solid-colored border edges (distinct non-bg colors).
- A central filled rectangle with 1px colored guide edges on two sides, colors matching the frame controls.

Rule:
- Rotate/mirror the object so its guide edges face the same sides as the matching frame controls.
- Crop off the 1px guide edges from aligned sides.

Key insight: Guide-edge colors encode target frame sides; the correct dihedral transform matches both simultaneously.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="9bbf930d" lines="61-72">
<description>
Move magenta particles along corridors formed by parallel colored bars.

Input structure:
- bg=7. Particles=6 in col 0. Colored bars on even rows form corridors on odd rows.

Rule:
- Particle goes straight through bg cells.
- When blocked by a colored cell, turns 90° into the perpendicular direction
  that has a bg cell flanked by same-color bars. Stops at grid edge or full blockage.

Key insight: Same-color parallel bars define the only valid corridors, so each particle follows a deterministic local trace.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="db695cfb" lines="8-19">
<description>
Connect same-color diagonal pixel pairs, then emit perpendicular rays from differently-colored pixels on those diagonals.

Input structure:
- bg = most common color. Line color (e.g. blue) has pixel pairs aligned on shared diagonals. Trigger color (e.g. magenta) pixels sit along those diagonals.

Rule:
- Draw diagonal lines connecting each pair of same-color aligned pixels.
- Where the diagonal crosses a trigger pixel, emit perpendicular diagonal rays (in the trigger's color) extending to the grid boundary.

Key insight: Trigger pixels act as beam-splitters — the perpendicular rays rotate 90° from the connecting diagonal's orientation.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="981571dc" lines="6-19">
<description>
Reconstruct a 4-way symmetric mosaic by filling missing (black) pixels.

Input structure:
- A colored grid exhibiting bilateral symmetry along both a vertical and a horizontal axis.
- Some pixels are erased to color 0 (black), creating gaps in the pattern.

Rule:
- Infer the vertical and horizontal reflection axes from the visible (non-zero) pixels.
- For each black pixel, recover its color from the mirrored position across either axis or both.
- When multiple mirrors yield candidates, use the majority color.

Key insight: The reflection axes are not given — they must be deduced from the surviving pixels by finding the axes that best preserve the existing symmetry.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="eval" puzzle_id="7491f3cf" lines="9-24">
<description>
Composite two pattern tiles into a fourth panel using flood-fill from an indicator pixel.

Input structure:
- Four 5×5 panels in one row, border-separated.
- Panel 0: cue pixels = wall boundary (8-connected, edge-to-edge, through center)
           + one 8-isolated indicator pixel.
- Panels 1–2: source patterns. Panel 3: blank (to fill).

Rule:
- Flood-fill (4-conn) from indicator, blocked by wall → inside region.
- Inside → non-bg from panel 1; outside → non-bg from panel 2.
- Wall → panel 1 if non-bg; panel 2 fallback when wall is 4-connected.

Key insight: The cue panel's isolated indicator and wall partition the tile into inside/outside masks that select between the two source panels.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="d35bdbdc" lines="120-134">
<description>
Transfer center colors between bordered sprites matched by color, keeping only path-connected ones.

Input structure:
- bg, grey (5) path network, and several small "bordered sprites" each having a border color and a single center pixel of a different color.
- Grey path forms a one-stroke trail under 8-connectivity; its start/end cells touch exactly two destination sprites via ports.

Rule:
- Destination sprites are determined by the grey trail's two endpoints.
- For each destination, find the source sprite whose border color equals the destination's center color.
- Replace the destination's center pixel with the source's center color.
- Delete all non-destination sprites; grey paths and destinations remain.

Key insight: Center color of each destination acts as a lookup key into the border colors of other sprites, enabling indirect color transfer.
</description>
</reference_puzzle>

你的任务：
以参考题为标杆，运用发散性思维进行创新扩展。请发挥创造力，构思全新且巧妙的推理谜题。用中文创造全新的 ARC 谜题描述。
创新方式包括但不限于：
**机制重组**：将多道参考题各自的核心机制提取出来，组合成一个新的多步管道，用到新的场景。
**机制变异**：取几道参考题的核心机制，以机制为基础进行创新并与其他多种机制进行创新重组，使解题路径发生质变而非量变。
**逆向构造**：取一道参考题的输出模式，思考"什么样的输入规则能产生类似的输出结构但需要完全不同的推理过程"。

严格使用下面的 XML 风格格式输出：

<new_puzzle>
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。
</new_puzzle>

请一次性生成 5 个 `<new_puzzle>`，若有较多合适的创新，也可以生成更多，不设上限，但不要输出额外解释。

注意事项：
1. 任务不应依赖大量先验知识或复杂计算，而应依赖对模式、结构与空间关系的直觉推理。
2. 关于"计算"的边界：
   - **允许使用**（这些都是人类能心算的视觉量）：数像素 / 孔洞 / 对象 / 邻居（≤10 量级）、比大小、按某个可见属性排序、奇偶判断、取最大 / 最小、用一个计数值作为另一个等量几何量（如 N 个像素 → 长度 N 的射线、面积 N → 边长 N）、判断接触 / 相邻 / 包含、粗略中心位置（簇的视觉中点）、"距离"作为"几步"而不是公式等人类能够直观感受计算的内容。
   - **禁止使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算 / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算都是禁止使用的。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息（例如颜色id和颜色亮度是禁止使用的）或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
4. 生成的问题描述旨在促进后续用于创建题目网格的代码生成。因此，对输入 / 输出网格的描述以及规则总结必须极其详细；语言风格应易于理解；对于特定对象（如箭头），必须在描述中明确其绘制方式。
5. 描述谜题时避免使用 0 或 1 这类颜色编号；应使用颜色名称，例如 "red"。可用颜色如下：Black、Blue、Red、Green、Yellow、Grey、Magenta、Orange、Azure、Maroon。
6. 5 个新题的核心推理机制应明显不同——即从观察样例到归纳规则的认知路径不同，而不仅仅是场景、对象或颜色不同。
7. 题目难度应来自机制的隐蔽性而非步骤数。用参考题作为难度标尺——如果你生成的题一个样例就能猜到规则，说明机制太透明，需要加深隐蔽性。
8. 优先从参考题的机制出发进行重组或推演，而非从物理隐喻出发。物理隐喻容易导致大量题目归结为一种模式，例如"沿方向移动到阻挡处停止"。
9. 题目的难度优先来自规则的**结构层数**，而不是单一步骤的复杂度。关于"结构层数"的判定标准：**只有解题者需要主动推理才能完成的步骤才算一层**。肉眼一扫即可获得的信息（如"找到缺口在哪一侧"、"数出像素个数"、"识别哪个对象最大"）不构成独立的推理层——它们只是变换的前置输入，不增加认知难度。真正的多层结构是指每一层都需要解题者发现一个非显然的规律。以两层结构为例：
   - ✗ 伪两层："找到缺口方向"→"把东西推到对面" — 第一步肉眼可见，实际只有一层推理。
   - ✓ 真两层："通过孔洞数量匹配颜色图例"→"用匹配到的颜色重绘对象" — 两步都需要非平凡的归纳。

   如果你发现自己写出的题只是"对每个对象，根据它的某个属性做某变换"的单步规则，请引入一个需要主动推理的前置步骤。**用堆叠 tie-break 规则或复杂数学公式来"加难度"是劣解；用多一层需要归纳的结构嵌套来"加难度"才是好解**。
10. 在符合ARC题目和提示词要求的基础上，鼓励尝试创新规则为多层结构的题目，不局限于两层，可以创新三层或四层结构的题目，但不可为了多结构层数违背要求。

生成后对生成的题执行以下四项审查（审查内容不需要保存到文件）：
1. 原创性
  新题不可与上述参考题相同或类似，不可只是替换颜色、方向、对象类型等表层要素。判断标准：新题的核心推理路径应与任何一道参考题不同。
2. 确定性
  规则完全确定，同一输入只能产生唯一输出，每一步操作都有明确定义。
3. 可实现性
  所有必要细节都已给出——对象如何生成、如何识别、如何变换、边界情况如何处理，读完描述后可以直接写代码。
4. ARC 合规性
  网格尺寸合理（≤30×30），使用颜色不超过 10 种，规则复杂度适中，人类能从 3-5 个样例中发现规律。没有使用上述注意事项第二条禁止使用的复杂计算。

