你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="2de01db2" lines="28-41">
<description>
Invert each row's native foreground color against the black background.

Input structure
- A black-background grid whose rows each have one native foreground color plus optional distractor colors.
- The native row color is the most frequent non-bg color in that row; if tied, its run starts leftmost among tied colors.

Rule
- For each row independently, turn pixels of the native row color into bg.
- Turn every other pixel in that row, including bg and distractor colors, into the native row color.

Key insight
- Distractor colors are not preserved objects; they mark cells that participate in the binary complement of the row's native color.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="eval" puzzle_id="7666fa5d" lines="12-26">
<description>
Fill the parallelogram-shaped region between parallel anti-diagonal line segments with red.

Input structure:
- bg = most common color.
- One non-bg color forms short diagonal segments, each lying on a single anti-diagonal (r+c = const).
- Segments come in parallel pairs bounding a region to be filled.

Rule:
- For each main-diagonal axis (r−c = const), collect anti-diagonal segments whose span covers that axis.
- Sort those segments by their anti-diagonal index (r+c).
- Fill all bg cells lying strictly between consecutive segment pairs along that axis with red (color 2).

Key insight: The two diagonal directions decouple — segments sit on anti-diagonals, and filling sweeps along main diagonals between them, forming parallelogram regions.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="b7256dcd" lines="18-32">
<description>
Transfer each colored marker to its adjacent neutral mask component, then erase the markers.

Input structure
- The background is the most common color.
- The neutral mask color is the unique most frequent foreground color.
- Other foreground colors are marker pixels that may touch neutral 4-connected mask components.
Rule
- Remove every marker pixel by recoloring it to the background.
- For each 4-connected neutral mask component, find marker colors touching it by edge adjacency.
- If a component touches one marker color, recolor the whole neutral component to that marker color.
- If a component touches no marker color, leave it neutral.
Key insight
- Marker pixels label neighboring neutral regions but are not part of the final colored objects.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="d749d46f" lines="32-51">
<description>
Pack each solid foreground rectangle twice: wide across the top and tall from the bottom.

Input structure
The grid has one background color and one foreground color; the foreground forms
separate solid rectangles, ordered from left to right.

Rule
Build a 10-row background canvas whose width is the packed width of the wide
versions of the rectangles. For each source rectangle, paint a top-aligned
rectangle of shape min(h,w) by max(h,w), with one background column between
consecutive rectangles. Then, starting again from the left edge, paint a
bottom-aligned rectangle of shape max(h,w) by min(h,w) for each source rectangle,
again separated by one background column.

Key insight
The input rectangles are not copied in their original orientations; each object
contributes one horizontal-facing and one vertical-facing normalized rectangle,
preserving left-to-right object order.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="eval" puzzle_id="16de56c4" lines="9-34">
<description>
Fill periodic progressions that shoot from a unique grid edge.

Input structure:
- bg = most common color.  Sparse non-bg dots of several colors.
- All repeating colors originate from the same grid edge (top, bottom,
  left, or right — the one with the most non-bg dots).
- Each repeating color has exactly 2 dots per shooting-direction line;
  their gap defines the step.
- Every non-bg dot lies on a shooting-direction line that contains a
  repeating color.

Rule:
- Determine the unique shooting edge; left/right → rows, top/bottom → columns.
- For each shooting-direction line, the repeating color has exactly 2 dots;
  step = gap between them, offset = first position mod step.
- A different-colored dot aligned to the same beat (pos % step == offset) is
  a companion: it absorbs the ray and becomes the fill color, limiting the
  fill range to the span between repeating and companion dots.
- An unaligned dot is a bystander and left untouched.
- Without a companion the repeating color fills the entire line.

Key insight: the shooting edge is identified by counting boundary dots;
the gap between the two marker dots is the period; a companion sharing the
modular class acts as a wall that stops the ray and dictates the paint color.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="34cfa167" lines="94-117">
<description>
Complete a four-corner framed rectangle from one decorated anchor rectangle and its diagonal mate.

Input structure:
- The background is the dominant color.
- Two same-color solid rectangular anchors of the same size sit on opposite
  diagonal corners, with the first anchor above-left of the second.
- The upper-left anchor has two one-cell-thick vertical control bars to its
  right and two one-cell-thick horizontal control bars below it.

Rule:
- Copy the anchor rectangle into the two missing corners of the implied
  larger rectangle.
- Repeat the right-side control bars horizontally between the left and right
  anchors, using their spacing as the period, and draw the outer top and
  bottom borders in the first right-control color.
- Repeat the below-side control bars vertically between the top and bottom
  anchors, using the same period, and draw the outer left and right borders
  in the first below-control color.

Key insight: The small control bars are the first two ticks of two periodic
frame patterns; the diagonal anchor supplies the opposite corner and the
missing two anchor copies.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="230f2e48" lines="19-38">
<description>
Fold each straight red rod at its black marker, leaving the gray-anchored side fixed.

Input structure:
- The background is the dominant color.
- Each foreground object is a straight horizontal or vertical 4-connected rod.
- Each rod contains many pixels of one line color, one endpoint marker, and one
  interior control marker.

Rule:
- For every rod, keep the segment from the endpoint marker through the control
  marker unchanged.
- Move the line-color tail on the far side of the control marker to a
  perpendicular ray starting next to that marker.
- Choose the perpendicular direction with more space to the grid edge.

Key insight:
- The endpoint marker identifies the fixed side of the rod, while the control
  marker is the hinge; the tail folds into the larger open half-plane.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="fd02da9e" lines="26-41">
<description>
Replace each colored corner marker with its corner-specific inward stencil in the same color.

Input structure
- The grid is 8x8 with a single dominant background color.
- Foreground pixels are marker pixels, each located at one of the four corners.
- A marker's color is carried to the stencil selected by its corner position.

Rule
- Erase each corner marker back to the background.
- Top corner markers become an inward 2x2 block one cell away from the top and side edges.
- Bottom corner markers become the mirrored three-pixel inward bent stencil shown by the bottom-corner examples.

Key insight
- The marker position selects the fixed stencil; the marker color only supplies the paint color, so multiple corners apply independently.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="5751f35e" lines="29-43">
<description>
Complete sparse nested rectangular color layers into solid rectangles.

Input structure
The background is the dominant color, and every foreground color marks one
axis-aligned rectangular layer; the layer bboxes have unique areas and are
nested from outermost to innermost.
Rule
Take each color's bbox as its completed rectangle, except that a child bbox
which already matches at least three sides of its parent's one-pixel inset
expands to that full inset; repaint the rectangles largest to smallest.
Key insight
The scattered pixels are samples of nested rectangular regions, so the
output is a clean redraw of those regions with inner colors overwriting
outer colors.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="e3721c99" lines="69-83">
<description>
Recolor grey shapes by matching their hole count to a color legend.

Input structure:
- bg = most common color; grey (5) sprites scattered across the grid.
- Colored legend shapes (non-bg, non-grey, at least 2×2, not touching the outer border) near grid edge,
  each with a distinct number of enclosed holes.

Rule:
- Each colored legend shape defines a mapping: hole count → color.
- For every grey sprite, count its holes (enclosed bg regions not touching the sprite's bounding-box border).
- Recolor the grey sprite to the matching legend color; erase grey distractors with unlisted hole counts.

Key insight: Shape identity is encoded by topology (number of holes), not geometry or size.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="16b78196" lines="6-20">
<description>
Assemble colored sprites into a central bar's notches, chaining successive pieces along fixed-width strips.

Input structure:
- bg = most common color. Largest monochrome component is a solid "bar" with boundary notches (bg holes opening to its bbox edge).
- Several smaller colored sprites scattered around the bar.

Rule:
- Each bar notch is filled by a root piece whose shape exactly covers the hole.
- The root defines a strip (column-span or row-span) for its chain.
- Successive pieces fill the bg cells inside the previous piece's bbox, locked to the same strip width.
- Multiple chains grow independently from different bar notches.

Key insight: The root piece's extent perpendicular to the chain direction locks the strip width for all subsequent pieces in that chain.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="0934a4d8" lines="75-90">
<description>
Reveal the rectangular patch hidden by the solid marker using the grid's reflection symmetries.

Input structure:
- A square colored pattern with one solid rectangular marker color occluding a patch.
- The visible pattern has horizontal and vertical reflection symmetry on a large lower/right strip.
- The top and left outer strips also expose transpose counterparts for cells not recoverable by those reflections.

Rule:
- Find the unique color whose full extent is a solid rectangle; this rectangle is the output region.
- Infer the reflected strip start as the smallest row/column offset whose lower/right strips are mirror-consistent, ignoring marker pixels.
- For each marker pixel, copy the unique non-marker value in its horizontal/vertical reflection orbit; if that orbit is entirely hidden in the outer strip, use the transposed top/left-strip counterpart.

Key insight:
The marker is not part of the pattern; it covers a missing rectangle whose cells are still determined by the visible mirror or transpose copies elsewhere in the grid.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="33067df9" lines="32-48">
<description>
Expand the sparse dot lattice into a 26x26 drawing with thick bars between adjacent matching dots.

Input structure
- The background is the dominant color.
- Foreground pixels lie on logical rows and columns separated by background spacer rows/columns.
- A logical row or column is present when it contains at least one foreground pixel.

Rule
- Expand every background spacer row/column to thickness 2.
- Expand the logical rows/columns equally so the full output is 26x26.
- Paint each foreground logical pixel as a solid rectangle of its color.
- Connect adjacent same-color logical pixels horizontally through the intervening spacer.
- Connect adjacent same-color logical pixels vertically only when neither endpoint is part of a horizontal same-color run.

Key insight
- The input is a sampled line drawing: horizontal same-color runs define the bars first, and only unclaimed same-color vertical neighbors are joined.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="409aa875" lines="68-81">
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

<reference_puzzle index="15" source="eval" puzzle_id="b0039139" lines="29-44">
<description>
Recolor a cropped template and repeat it according to a component-count stencil.

Input structure:
- Solid blue (1) separator rows or columns split the grid into four sections.
- Section 1 is a black-background shape template; section 2 is a component-count stencil.
- Sections 3 and 4 are solid color swatches for output foreground and background.

Rule:
- Crop section 1 to the bbox of its non-black pixels and recolor it with the two swatches.
- Count 4-connected non-black components in section 2.
- Repeat the recolored crop that many times, inserting one background-colored row/column between copies.

Key insight: The second section encodes only the repeat count; the output orientation
is inherited from whether the input separators are columns or rows.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="7b3084d4" lines="253-266">
<description>
Assemble colored rectangular sprites into a single tiled rectangle, anchored by a marker pixel.

Input structure:
- bg = most common color. Several solid-colored rectangular sprites scattered on the grid.
- One sprite has a unique single-pixel marker (different color) at one corner — this is the anchor.

Rule:
- Identify the anchor sprite (contains the marker pixel) and the remaining pieces.
- Rotate the anchor so the marker pixel lands at position (0,0) of the output.
- Pack all remaining pieces (allowing 90° rotations) to tile a complete rectangle with no gaps.

Key insight: The lone marker pixel picks the anchor and fixes the output orientation — it must sit at the top-left corner.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="17b866bd" lines="49-67">
<description>
Move each marker from a lattice corner gap into the adjacent enclosed 12-pixel cavity.

Input structure:
- The grid is a repeated teal lattice around black 12-pixel cavities.
- A marker is a non-black pixel placed at the northwest corner gap of one
  complete 5x5 lattice tile; the marker may itself be teal.

Rule:
- For every marked northwest corner gap, clear that marker to black.
- Fill the 12 black pixels in the tile immediately down-right of the marker
  with the marker color.
- Leave the rest of the lattice unchanged.

Key insight:
- The apparent isolated markers are control pixels for the adjacent cavity;
  teal markers are recognized as extra teal pixels occupying a gap that the
  periodic lattice template says should be black.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="271d71e2" lines="9-22">
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

<reference_puzzle index="19" source="eval" puzzle_id="b9e38dc0" lines="7-18">
<description>
Fill the region enclosed by a wall sprite and radiate color outward through its opening.

Input structure:
- bg = most common color. Wall = most frequent non-bg color, forming a partial enclosure (e.g. U-shape).
- A fill-color pixel sits inside the enclosed area. Other non-bg pixels may act as obstacles.
Rule:
- Flood-fill the enclosed area with the fill color, radiating outward through the wall's open end.
- Obstacles cast "shadows" along the radiate direction, blocking fill behind them.

Key insight: The wall endpoints define the opening direction; fill radiates outward through it, and obstacles block propagation along that same axis.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="342dd610" lines="18-36">
<description>
Move every foreground pixel by the fixed offset indicated by its color.

Input structure:
- The grid has a unique dominant background color.
- The foreground is a sparse set of colored pixels.
- Each foreground color is one of the trained motion colors: orange, red,
  maroon, or blue.

Rule:
- Clear the canvas to the background color.
- Translate orange pixels up 2 cells, red pixels left 2 cells, maroon
  pixels down 2 cells, and blue pixels right 1 cell.
- Preserve each moved pixel's original color.

Key insight:
The foreground color is the motion instruction; position only determines
where that color-specific displacement lands.
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

请一次性生成5个`<new_puzzle>`，若有较多合适的创新，也可以生成更多，不设上限，但不要输出额外解释。

注意事项：
1. 任务不应依赖大量先验知识或复杂计算，而应依赖对模式、结构与空间关系的直觉推理。
2. 关于"计算"的边界：
   - **允许使用**（这些都是人类能心算的视觉量）：数像素 / 孔洞 / 对象 / 邻居（≤10 量级）、比大小、按某个可见属性排序、奇偶判断、取最大 / 最小、用一个计数值作为另一个等量几何量（如 N 个像素 → 长度 N 的射线、面积 N → 边长 N）、判断接触 / 相邻 / 包含、粗略中心位置（簇的视觉中点）、"距离"作为"几步"而不是公式等人类能够直观感受计算的内容。
   - **禁止使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算（取模） / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算都是禁止使用的。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息（例如颜色id和颜色亮度是禁止使用的）或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
4. 生成的问题描述旨在促进后续用于创建题目网格的代码生成。因此，对输入 / 输出网格的描述以及规则总结必须极其详细；语言风格应易于理解；对于特定对象（如箭头），必须在描述中明确其绘制方式。
5. 描述谜题时避免使用 0 或 1 这类颜色编号；应使用颜色名称，例如 "red"。可用颜色如下：Black、Blue、Red、Green、Yellow、Grey、Magenta、Orange、Azure、Maroon。
6. 新题的核心推理机制应明显不同——即从观察样例到归纳规则的认知路径不同，而不仅仅是场景、对象或颜色不同。
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

