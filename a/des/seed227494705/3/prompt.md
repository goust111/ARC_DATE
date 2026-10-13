你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="a25697e4" lines="9-23">
<description>
Dock movable tools into border-anchored frames by filling cavities.

Input structure:
- bg = most common color.
- Border-touching components cluster into anchor frames with interior bg cavities.
- Non-border 2-color components are movable tools.

Rule:
- Match each tool to a frame whose cavity size equals one of its color-region counts.
- Rotate/flip the tool so one color fills the cavity exactly; the other stays outside.
- If the docking orientation uses a reflection, swap the tool's two colors.

Key insight: Cavity size determines tool–frame pairing; reflection triggers color swap.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="30f42897" lines="51-66">
<description>
Repeat the single colored border segment around the rectangular border, leaving one segment-length gap between copies.

Input structure:
- The background is the dominant color.
- There is exactly one foreground color.
- All foreground pixels form one contiguous segment on the rectangular border, possibly crossing the cycle start.

Rule:
- Unwrap the grid border clockwise into a cyclic path.
- Let n be the length of the foreground segment.
- Copy the segment with the same color every 2*n border positions around the cycle.

Key insight:
- The foreground segment is not extended by rows or columns directly; it is a motif on the perimeter cycle, repeated with an equal-sized gap.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="c4d1a9ae" lines="28-48">
<description>
Fill each vertical foreground band's background cells with the color of the next band to its right.

Input structure
- The dominant color is the background.
- Each other color occupies a separate vertical band defined by that color's
  minimum and maximum columns.
- Within a band, all cells are either that band's color or background.

Rule
- Order the bands from left to right.
- In each band, replace background pixels with the color of the next band in
  that order, wrapping from the rightmost band back to the leftmost.
- Preserve all existing foreground pixels and all background pixels outside
  the bands.

Key insight
- The column bands form a cyclic palette legend: a band's gaps are completed
  by borrowing the color from its right-hand neighbor, not by changing the
  band's existing pixels.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="60d73be6" lines="22-39">
<description>
Complete the sparse colored marker pattern by reflecting it across a full-row/full-column cross.

Input structure
The grid has a dominant background and one non-background cross color that
fills exactly one complete row and one complete column. All other foreground
pixels are sparse markers away from that cross.

Rule
Keep the input unchanged, and for each marker pixel copy its color to the
positions obtained by reflecting it across the cross row, across the cross
column, and across both axes. Reflected positions outside the grid are
clipped.

Key insight
The full cross is not a copied object; it is the pair of mirror axes for
completing the quadrant pattern.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="ad3b40cf" lines="72-87">
<description>
Mirror the smallest non-axis foreground color across the full straight axis.

Input structure
- The background is the dominant color.
- One foreground color forms exactly one full row, column, main diagonal, or anti-diagonal reflection axis.
- The remaining foreground colors include a unique least-populated color to copy.

Rule
- Preserve the input grid.
- Select every pixel of the unique smallest non-axis foreground color.
- Reflect those pixels across the full straight axis and paint the reflected positions with the same color.

Key insight
- The axis color marks the mirror line, while the color population chooses which set of objects is duplicated.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="eval" puzzle_id="62593bfd" lines="32-46">
<description>
Sort objects to top or bottom edge by whether their shape (up to rotation/flip) is shared.

Input structure:
- bg = most common color.
- Several multicolor connected components (8-connected) scattered across the grid.

Rule:
- Compute each object's canonical shape (binary mask, invariant to rotation and flip).
- Objects whose shape appears in two or more components → translate to TOP edge (row 0).
- Objects whose shape is unique → translate to BOTTOM edge (row H−h).
- Horizontal position (left column) is preserved; only vertical translation changes.

Key insight: The sorting criterion is shape frequency (rotation-invariant), not area — duplicated shapes go up, unique shapes go down.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="95755ff2" lines="45-60">
<description>
Paint straight rays from colored side markers into the hollow diamond frame until an interior foreground pixel blocks each ray.

Input structure
- The background contains one complete one-pixel-thick diamond frame.
- Colored marker pixels outside the diamond sit on the top, bottom, left, or right side of the frame's bounding box.
- Any non-background, non-frame pixels inside the diamond are blockers that remain unchanged.

Rule
- Each outside marker emits inward perpendicular to its side of the diamond bounding box.
- Along that row or column, paint only background pixels strictly inside the diamond with the marker color.
- Stop a ray when it reaches an existing interior foreground pixel; keep the frame and all original colored pixels unchanged.

Key insight
- The diamond frame defines a strict interior, while the side markers are column or row emitters whose color fills the visible interior interval until blocked.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="eval" puzzle_id="8e5c0c38" lines="9-21">
<description>
Clean each color layer by enforcing vertical mirror symmetry.

Input structure:
- bg = background. Several non-bg colors, each forming a nearly-symmetric pattern about some vertical axis.
- Some pixels in each color are "noise" that breaks the symmetry.

Rule:
- For each non-bg color, find the vertical axis that best explains its pixel layout.
- Remove any pixel that lacks a same-color mirror counterpart across that axis.

Key insight: Each color has its own independent symmetry axis; the task is per-color denoising, not whole-grid symmetry.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="aaecdb9a" lines="10-26">
<description>
Convert scattered foreground objects into bottom-aligned count bars in a fixed five-color order.

Input structure
The grid has a dominant background color and foreground objects in the task
palette: grey, red, teal, maroon, and pink. Objects are monochromatic
8-connected components; some palette colors may be absent.

Rule
For each palette color in order grey, red, teal, maroon, pink, count its
8-connected objects. Create a background grid with five columns and height
equal to the largest count, then fill each column upward from the bottom
with its palette color for that count.

Key insight
The output does not preserve object shapes or positions; it is a histogram
of per-color 8-connected object counts, using the learned palette order.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="16b78196" lines="6-20">
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

<reference_puzzle index="11" source="eval" puzzle_id="1ae2feb7" lines="30-45">
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

<reference_puzzle index="12" source="train" puzzle_id="edcc2ff0" lines="47-63">
<description>
Keep only the lower-panel colors named by the top markers and expand each marker into an object-count bar.

Input structure
A black instruction strip sits above a non-black rectangular scene panel.  The strip contains one-cell markers
in the first column, and each marker color is a candidate object color in the lower panel.  The lower panel has
a unique dominant background color and scattered 4-connected foreground objects.

Rule
In the lower panel, preserve objects whose colors appear as top markers and recolor every other foreground
object to the panel background.  In the instruction strip, replace each marker by a horizontal run from column 0
whose length is the number of 4-connected lower-panel objects of that marker color; erase markers with count 0.

Key insight
The top strip is a color filter plus a compact count summary: marker colors select which lower-panel objects
survive, and their bars report the number of surviving objects by color.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="305b1341" lines="22-37">
<description>
Expand each sparse color object into a filled rectangular region using its legend-paired fill color.

Input structure
A background-color grid has a two-column legend in columns 0 and 1, starting at row 0: each row gives
a source color in the left column and that source color's fill color in the right column.  Outside the
legend, the foreground pixels are sparse dot objects whose colors are exactly the legend source colors.

Rule
Clear the legend.  For each source color, take the bbox of all of its non-legend dots, expand that bbox
by one pixel in every direction, and fill background cells inside the expanded rectangle with the paired
legend fill color while preserving every original non-legend dot.

Key insight
The legend order is also the layer order: earlier legend rows are in front, so fill rectangles are painted
from the last legend row back to the first and higher-priority fills overwrite lower-priority fills.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="6165ea8f" lines="74-90">
<description>
Build a color-labeled shape-equivalence table from the objects listed on the right edge.

Input structure:
- The background contains one main 4-connected object for each foreground color.
- The rightmost column has one singleton pixel of each foreground color, listed top-to-bottom.
- Those singleton pixels define the row and column order of the output table.

Rule:
- Normalize each main object to its tight mask and compare masks up to rotation or reflection.
- Draw 2x2 color labels across the top and down the left, separated by one background row/column.
- For each off-diagonal table cell, paint a 2x2 red block when the two labeled objects have the same shape under rotation/reflection, otherwise paint a 2x2 grey block.
- Leave diagonal cells and all separators as background.

Key insight:
- Object positions and original orientations are distractors; the relation is dihedral shape equivalence, while the right-edge legend fixes the table order.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="a47bf94d" lines="178-189">
<description>
Transmit colored tokens along wire networks, converting each to its complementary 3×3 pattern.

Input structure:
- bg = background. Non-bg cells form wire networks and 3×3 tokens (solid blocks or X/diagonal patterns) near wire endpoints.

Rule:
- Every solid 3×3 token is replaced in-place by a diamond (center + 4 cardinal neighbors).
- Each token near a wire endpoint sends its complement to the opposite endpoint: solid emits X, X emits diamond.

Key insight: Token shape cycles solid → X → diamond; wires route each complement to the far end.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="8886d717" lines="43-63">
<description>
Repair markers embedded in the main object and copy the remaining markers one pixel toward the solid edge.

Input structure:
- A mostly background grid contains one solid non-background edge line.
- Apart from that edge and the background, there is one large main object
  color and a smaller marker color.
- Each marker is either embedded in the main object, touching it on at
  least three 4-neighbor sides, or isolated on the background.

Rule:
- Embedded markers are recolored to the main object color.
- Isolated background markers remain in place and also paint the adjacent
  background pixel in the direction of the solid edge line.
- All other pixels are unchanged.

Key insight:
The full edge line is a direction cue: it tells which way the background
markers extend, while markers surrounded by the main object are simply
defects to be filled.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="9473c6fb" lines="10-25">
<description>
Recolor the sparse foreground pixels in scan order along their longer spread, cycling red, teal, and grey.

Input structure:
- The dominant color is the background.
- Every foreground object is a single isolated 4-connected pixel.
- The foreground pixel set has a rectangular bbox whose row or column extent defines the scan direction.

Rule:
- Keep the background unchanged.
- If the foreground bbox is at least as tall as it is wide, order foreground pixels row-major; otherwise order them column-major.
- Recolor the ordered pixels by repeating red, teal, grey from the first pixel onward.

Key insight:
- The original foreground colors are only markers; the output color depends on each marker's ordinal position in the dominant scan direction.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="db695cfb" lines="8-19">
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

<reference_puzzle index="19" source="train" puzzle_id="f341894c" lines="63-78">
<description>
Reorient each two-pixel blue/pink segment so it reads pink then blue when viewed from its border marker.

Input structure
- The background is dominant, with one border-only marker color and blue/pink foreground segments.
- Every marker lies on the outer border and points inward along its row or column.
- Along exactly one inward ray from each marker is a single adjacent two-pixel blue/pink segment.

Rule
- For each marker, look inward perpendicular to the border side containing the marker.
- Rewrite the two blue/pink pixels on that ray so the nearer one is pink and the farther one is blue.
- Keep marker pixels, background pixels, and all other pixels unchanged.

Key insight
- The marker gives a local viewing direction; the segment's required order is relative to that marker, not to global row or column order.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="b7955b3c" lines="386-403">
<description>
Replace gray plus-shaped damage masks with the hidden colors of the rectilinear mosaic.

Input structure:
- Color 8 appears only as isolated 3x3 plus masks.
- The non-mask pixels are visible parts of a clean grid made from horizontal
  and vertical color regions.

Rule:
- Treat every color-8 pixel as unknown damage and leave all other pixels fixed.
- First copy unknown pixels from exact matching parallel rows or columns.
- Then fill any remaining mask cells from opposite-side continuations and the
  better-supported local straight edge.

Key insight:
- The gray pluses are occluders, not objects; the underlying colors are
  determined by repeated line templates and nearby axis-aligned boundaries.
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

