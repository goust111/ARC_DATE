你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="80a900e0" lines="69-82">
<description>
Extend diagonal rays outward from outer-border segments adjacent to diamond shapes on a checkerboard.

Input structure:
- Checkerboard bg (two most common colors).
- Hollow diamond outlines (8-pixel) in a non-bg color (inner border).
- Short diagonal line segments in another non-bg color adjacent to each diamond edge (outer border).

Rule:
- For each outer-border segment, determine the outward direction (perpendicular to the segment, away from the diamond).
- From each endpoint of the segment, cast a diagonal ray outward, painting over bg pixels with the segment's color until hitting a non-bg pixel or the grid edge.

Key insight: Rays emit only from the two endpoints of each outer-border segment, not every pixel, and travel diagonally away from the diamond.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="eval" puzzle_id="64efde09" lines="68-83">
<description>
Copy colored control-ray ports from one rectangular sprite orientation to every matching transformed sprite.

Input structure:
- bg is the most common color.
- Several dense 2-by-N or N-by-2 rectangular sprites are dihedral transforms of one template.
- Isolated single-pixel control markers sit on rays projected outward from the dominant long edge of some sprites.

Rule:
- For each sprite, the active edge is the long side whose cells have the strongest same-color majority.
- A control marker seen first along an outward ray from an active-edge cell assigns its color to that template cell.
- On every transformed sprite, draw that assigned color outward from the corresponding active-edge cell through bg cells until blocked or out of bounds.

Key insight:
- The marker positions are not absolute; they label cells of a shared rectangular template, then the labels follow each sprite's rotation/reflection.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="97d7923e" lines="81-94">
<description>
Recolor vertical bars by legend-driven rank selection.

Input structure:
- bg = background. Top-left: monochrome blocks touching row 0; each has color K and height k.
- Main grid: vertical bars capped top and bottom by a single color, with solid-color interiors.

Rule:
- For each legend entry (K, k), find vertical bars capped by K.
- Rank bars by interior length (longest = rank 1).
- Recolor the rank-k bar's interior to K.

Key insight: Legend height is a rank index into length-sorted bars, not a literal size.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="20270e3b" lines="7-21">
<description>
Connect two sprites using matching connector pixels, then tight-crop.

Input structure:
- Background color `bg`, sprite color `obj`, connector color `ctrl` (small, 2 components).
- Two main foreground sprites, plus possibly tiny detached pixels belonging to one sprite.

Rule:
- Identify the **base** sprite (closest to grid top-left).
- Translate the other sprite so its connector sits one cell above the base's connector.
- Stamp the plug onto the base; remove connectors (recolor `ctrl` to `obj`).
- Output is the tight crop around all non-background pixels.

Key insight: The connector pixels define the unique docking point and direction.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="eval" puzzle_id="fc7cae8d" lines="9-21">
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

<reference_puzzle index="6" source="eval" puzzle_id="1ae2feb7" lines="30-45">
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

<reference_puzzle index="7" source="eval" puzzle_id="bf45cf4b" lines="7-19">
<description>
Scale up a monochrome control sprite by stamping a polychrome tile at each filled pixel.

Input structure:
- bg = background color.
- Two foreground objects (8-connected): a monochrome rectangular sprite with bg holes (control), and a polychrome square tile (block).

Rule:
- For each cell in the control sprite's bbox: if filled, stamp a copy of the block tile; if bg, leave a tile-sized bg region.
- Output grid dimensions are (control_h × tile_h) by (control_w × tile_w).

Key insight: The control sprite acts as a binary mask that gets "pixel-scaled" by the tile — each pixel expands to a full tile copy or empty space.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="271d71e2" lines="9-22">
<description>
Move bordered containers toward same-colored destination markers, charging gray cells to orange.

Input structure:
- bg = 6 (magenta). Black-bordered (0) rectangles with gray (5) / orange (7) interiors.
- A head sprite (9) sits adjacent to one container edge; a matching destination sprite lies along the movement axis.

Rule:
- Each container translates toward its destination along the head-to-dest axis.
- Steps = min(head-to-dest distance, number of gray interior cells).
- Gray cells become orange from the head-facing side inward, one per step.

Key insight: Gray cell count caps movement — a container stops short if it runs out of gray cells to charge.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="eval" puzzle_id="409aa875" lines="68-81">
<description>
Fire projectiles from 3-pixel triangle sprites; mark collisions with maroon or blue.

Input structure:
- bg = most common color. Non-bg 3-pixel sprites shaped as axis-aligned L's or diagonal V's.

Rule:
- Each triangle has an apex and an emission direction determined by its shape: L-shapes fire diagonally away from the missing corner; V-shapes fire along the axis away from the open side.
- A projectile lands exactly 5 cells from the apex in the emission direction.
- If it hits an existing sprite, that entire sprite is recolored to maroon (9).
- If two projectiles land on the same empty cell, that cell becomes blue (1); a lone projectile on empty bg paints maroon (9).

Key insight: The triangle's orientation encodes both direction and target; dual-projectile collisions on open bg produce blue, not maroon.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="898e7135" lines="7-20">
<description>
Inflate a frame's hole-layout and fill each hole with its matching piece.

Input structure:
- bg = most common color. Largest component is a monochrome frame enclosing bg holes.
- Outside pieces composed of k×k blocks; k = side of smallest solid square piece.

Rule:
- Extract connected hole regions inside the frame as slots.
- Match each piece to a slot by comparing block-level shapes under rotation/flip.
- Output the layout scaled by k, with pieces placed in slots; frame color as background.

Key insight: Pieces are k×k-block-scaled versions of holes; downsampling reveals shape matches under all 8 orientations.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="e12f9a14" lines="138-152">
<description>
Emit diagonal beams from framed 2x2 blocks through frame gaps; beams interact on collision.

Input structure:
- bg = most common color. Wall color (e.g. green) forms 4x4 frames around 2x2 monochrome blocks.
- Each frame has one or more gap pixels where the wall is missing.

Rule:
- Each gap emits a beam of the inner block's color, traveling diagonally outward (direction set by gap position relative to the block).
- Beams paint every cell they cross until they hit the grid edge or collide.
- 90° collision: both beams turn 45° toward their bisector direction; the intersection cell is left blank (1-pixel gap).
- 45° collision: beams merge into a zigzag (alternating cardinal + diagonal steps) with no gap at the meeting point.

Key insight: Two distinct collision rules — 90° produces a gap and single-pixel diagonal lines; 45° produces no gap and thicker zigzag bands.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="3e6067c3" lines="7-20">
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

<reference_puzzle index="13" source="eval" puzzle_id="9aaea919" lines="38-51">
<description>
Transfer sprites from red-marked lanes to green-marked lanes, recoloring and stacking.

Input structure:
- bg = most common color. Red (2) and green (3) thin bars define vertical lanes.
- Each lane contains repeated identical monochrome sprites in an evenly spaced vertical grid.

Rule:
- Each red lane's sprites turn grey (5); their count transfers to the nearest green lane.
- New sprites fill empty lattice slots bottom-up, using the source shape recolored to the destination lane's sprite color.
- All marker bars are removed.

Key insight: Sprite count is conserved across lanes — red sprites become grey, and equally many new sprites stack into the green lane's regular grid.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="35ab12c3" lines="73-88">
<description>
Complete outlined shapes and carry attached shadow pixels along their edges.

Input structure:
- bg = background. Each active color marks vertices of an 8-direction aligned outline.
- A start vertex can have adjacent non-bg, different-colored shadow pixels.
- Other colored pixels may be distractors unless they lie on valid drawable edges.

Rule:
- For each color, order its pixels by polar angle around their centroid to form a cycle.
- Draw only horizontal, vertical, or 45-degree edges whose interiors are clear.
- From the chosen start vertex, copy adjacent shadow offsets along each drawable edge
  when the parallel path is in bounds and clear.

Key insight: The main outline is reconstructed first, then the start vertex's neighbor offsets define parallel shadow strokes.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="7491f3cf" lines="9-24">
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

<reference_puzzle index="16" source="eval" puzzle_id="abc82100" lines="70-86">
<description>
Use local source-target-pattern examples to replace matching source pixels.

Input structure:
- Black background with isolated colored pixels and teal (8) pattern components.
- A rule example is a 4-neighbor colored pair source->target whose continued
  direction reaches an anchor touching a teal pattern component.

Rule:
- For each rule example, copy that teal component onto every other pixel of
  source.color, recoloring the copied component to target.color.
- Translation maps the rule anchor onto the source pixel being replaced.
- Remove teal components and all pixels used by rule examples.

Key insight: The two adjacent colored pixels define the color substitution and
the anchor point; the nearby teal component is the stencil that gets recolored.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="d35bdbdc" lines="120-134">
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

<reference_puzzle index="18" source="eval" puzzle_id="135a2760" lines="6-19">
<description>
Repair noisy periodic patterns inside frame-delimited grid regions.

Input structure:
- Frame color (most common) forms full rows/cols, dividing the grid into rectangular regions.
- bg (second most common) fills empty space; each region contains a small colored tile repeated periodically with a few corrupted pixels.

Rule:
- For each region, find the bounding box of non-bg content.
- Infer the smallest repeating tile that explains the pattern (majority vote per residue position), requiring it repeat at least 2x per axis.
- Replace the region with the clean tiled version.

Key insight: Tile dimensions are unknown and must be inferred; prefer the smallest tile with genuine repetition over one that memorizes the entire region.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="eval" puzzle_id="53fb4810" lines="105-119">
<description>
Extend marker tiles outward from '+' shapes by repeating them to the grid edge.

Input structure:
- bg = most common color.
- A dominant non-bg color forms several '+'-shaped objects.
- Small colored patches (1–few cells) sit adjacent to each arm of a '+'.

Rule:
- For each marker tile touching a '+', determine which side of the '+' it sits on.
- Project the tile away from the '+' in that direction, repeating the tile pattern periodically until the grid boundary.
- '+' pixels are never overwritten.

Key insight: Each marker's attachment side determines its extension direction; the pattern repeats periodically, not as a single copy.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="36a08778" lines="8-21">
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
   - **禁止使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算（取模） / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算都是禁止使用的。
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

