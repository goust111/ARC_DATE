你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="9ba4a9aa" lines="75-90">
<description>
Return the ring object reached by following the guide path from the checkerboard object.

Input structure
- The input has one dominant background color, one 3x3 checkerboard object,
  several 3x3 ring objects, and sparse guide pixels on a two-cell lattice.

Rule
- Find the checkerboard object and its unique adjacent guide pixel.
- Follow guide pixels of that color through two-cell horizontal/vertical steps.
- Output the unique 3x3 ring object adjacent to that connected guide path.

Key insight
- The checkerboard object is the source, and the color of its attached guide
  pixel selects the path whose far endpoint points at the answer ring.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="f3e14006" lines="21-39">
<description>
Build the woven rectangle determined by the two marked cells on each crossed bar.

Input structure
A black-background grid contains one long horizontal bar and one long vertical bar.
Each bar has a dominant base color and exactly two same-colored marker pixels.

Rule
The horizontal marker pair gives the output column span.  From the horizontal bar
row, walk toward the farther vertical marker; rows and columns alternate every
other cell.  Even step/even offset cells use the horizontal marker color until
the nearer vertical marker is reached, then use the vertical marker color.
Odd row steps use the horizontal base on marker columns and the original
crossing color on spacer columns.  Spacer columns on even row steps use the
vertical base, except at the switch row where they use the crossing color.

Key insight
The original cross cell records which base color wins at base/base lattice
crossings; the nearer vertical marker is the color-switch boundary.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="7c66cb00" lines="9-24">
<description>
Dock two-tone sprites into color-matched horizontal bars with a color swap.

Input structure:
- bg = most common color.
- Top area: small two-tone sprites (outer color + inner color).
- Below: horizontal bars spanning most of the row, each with a fill color and border color on edges.

Rule:
- Match each sprite to the bar whose fill color equals the sprite's outer color.
- Place the sprite at its original x-position, aligned to the bar's bottom.
- Recolor: sprite outer → bar border, sprite inner → bar fill.
- Erase original sprites from the top area.

Key insight: Color matching is inverted — the sprite's dominant color identifies its target bar, then gets swapped to the bar's accent color.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="16b78196" lines="6-20">
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

<reference_puzzle index="5" source="train" puzzle_id="4a1cacc2" lines="8-23">
<description>
Fill the rectangle from the lone foreground pixel to its nearest grid corner.

Input structure
- The grid is a uniform background field with exactly one foreground pixel.
- The foreground pixel's color is the color used for all added pixels.

Rule
- Find the grid corner with the smallest Manhattan distance to the foreground pixel.
- Recolor every pixel in the inclusive axis-aligned rectangle bounded by that
  corner and the foreground pixel to the foreground color.

Key insight
- The single pixel is one corner of the rectangle; the opposite corner is the
  nearest corner of the whole canvas.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="eval" puzzle_id="78332cb0" lines="10-23">
<description>
Reorder 5×5 blocks into a connected strip by matching edge foreground pixels.

Input structure:
- bg = background; sep = divider color (full row/column lines).
- Several 5×5 sprite blocks in a grid, separated by sep lines.

Rule:
- Two blocks "connect" when their facing edges share foreground pixels at the same offset (e.g., A's right edge ∩ B's left edge).
- Chain all blocks in the orientation (horizontal or vertical) that admits a complete sequence.
- Output is a 1×N strip of blocks in chain order, separated by sep lines.

Key insight: Edge-contact overlap determines both block ordering and output orientation.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="e45ef808" lines="31-46">
<description>
Mark the shallowest and deepest columns of a solid terrain boundary with vertical colored guide lines.

Input structure:
- The first row is a solid empty separator.
- Below it, each column consists of upper fill cells followed by a contiguous suffix of terrain cells.
- The bottom row is all terrain, so every column has a well-defined first terrain row.

Rule:
- Find the column whose first terrain row is latest; it is unique and its upper fill run becomes maroon.
- Find the rightmost column whose first terrain row is earliest; its upper fill run becomes yellow.
- Preserve the empty separator, terrain cells, and all other upper fill cells.

Key insight:
- The task is a column-height summary of the terrain profile: color the deepest upper run and the highest terrain peak.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="4df5b0ae" lines="27-41">
<description>
Pack the colored sprites along the bottom edge from smallest to largest.

Input structure
The grid has a dominant background color and several monochrome 4-connected
foreground sprites; a full-grid one-cell frame may surround the scene.

Rule
Remove any full-grid rectangular frame, sort the remaining sprites by pixel
area from smallest to largest, and paste their bounding-box sprites left to
right with each sprite bottom-aligned.

Key insight
The original locations do not determine the final slots; each sprite's area
gives the packing order, while its shape and color are preserved.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="e048c9ed" lines="11-26">
<description>
Add a quantity-color label in the top marker column for each horizontal bar.

Input structure
- The grid has a black background and one non-background marker pixel in the top row.
- Below the marker row, every non-empty source row contains one solid monochrome horizontal bar to the left of the marker column.
- Source rows are staggered every other row.

Rule
- Copy the input grid.
- For each source bar, read its quantity as the bar length.
- Paint the quantity color in the marker column on the same row as the bar:
  2->1, 3->4, 4->9, 5->6.

Key insight
- The grey marker supplies the result column, while each bar supplies a small quantity with a fixed color code.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="45a5af55" lines="5-18">
<description>
Build nested square frames from a stack of horizontal color stripes.

Input structure:
- Full-width horizontal stripes of distinct non-bg colors, stacked vertically.
- Each stripe has uniform color and variable height (1-4 rows).

Rule:
- Read stripes bottom to top.
- Bottom stripe becomes the center solid square with side length = stripe height.
- Each subsequent stripe wraps the current shape with a square frame of thickness = stripe height, in that stripe's color.

Key insight: Stripe height doubles as both frame thickness and (for the innermost) center-square side length; output dimensions are fully determined by the stripe heights.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="f8cc533f" lines="10-25">
<description>
Complete every colored sprite by applying the union of all aligned partial sprite masks.

Input structure:
- The dominant color is the background.
- Each non-background color appears as a separated partial copy of one common sprite mask.
- The copies share a top-left-aligned sprite window; some copies may be missing lower or right edge pixels.

Rule:
- Crop each same-color mask to its bounding box.
- Pad the crops to the maximum observed height and width, aligned at the top-left, and OR them into one complete template.
- Fill that template at every sprite location using the sprite's own color.

Key insight:
- No single input sprite is complete, but their missing cells are complementary, so the aligned union reveals the full mask.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="7666fa5d" lines="12-26">
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

<reference_puzzle index="13" source="train" puzzle_id="ac0c2ac3" lines="7-19">
<description>
Fill each centered square ring with the color of the marker at that ring distance.

Input structure
The input is an odd square grid with a dominant background color and one
non-background marker for every Chebyshev distance from the grid center.
Rule
The output replaces the background with concentric square rings centered on
the middle cell; every cell at distance d from the center takes the color of
the unique marker whose distance from the center is d.
Key insight
The scattered markers are not objects to move; their Chebyshev distances
form a palette for the centered square rings.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="291dc1e1" lines="7-22">
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

<reference_puzzle index="15" source="eval" puzzle_id="4c7dc4dd" lines="7-21">
<description>
Learn a transformation rule from a training pair of framed sprites, then apply it to a test sprite.

Input structure:
- Grid contains several colored square frames (hollow borders of a single color).
- Each frame's interior holds a small pattern sprite; one frame's interior is blank (all black).
- Two frames form a training pair (their interiors share the same color palette); one remaining frame is the test case.

Rule:
- Identify the training pair by matching interior color sets; the other non-blank frame is the test input.
- Infer the rule mapping source→destination (e.g. reflection, color inversion, gap filling, endpoint extraction, or checkerboard complement).
- Apply that same rule to the test sprite; output the transformed interior.

Key insight: The puzzle is an analogy — "A is to B as C is to ?" — where you must discover the visual rule from one example pair and generalize it.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="689c358e" lines="72-88">
<description>
Mark each arrow-like sprite's pointed frame endpoint with its color and the opposite endpoint black.

Input structure:
- The grid has a one-pixel solid rectangular frame around a dominant interior background.
- Inside the frame are monochrome six-pixel sprites shaped as asymmetric T arrows.
- Each sprite has a length-4 bar and a length-3 crossbar one cell closer to the pointed end.

Rule:
- For each sprite, follow its bar axis from the shorter end to the frame.
- Paint that frame endpoint with the sprite color.
- Paint the opposite endpoint on the same row or column black.
- Leave the interior background, sprites, and all other frame pixels unchanged.

Key insight:
- The direction is encoded by the crossbar's off-center position: the nearer end is the pointing end.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="58f5dbd5" lines="20-34">
<description>
Carve color-matched motifs into solid blocks and pack them into a compact grid.

Input structure:
- bg = most common color.
- Several solid 5×5 monochrome blocks, each a distinct non-bg color.
- For each block color, a small same-colored motif sprite elsewhere on the grid.

Rule:
- Match each motif to its same-colored 5×5 block.
- Center the motif within the block and recolor those cells to bg (punching holes).
- Pack all carved blocks into an output grid preserving their original row/column order, separated by 1-cell gaps and a 1-cell outer border.

Key insight: The motif acts as a stencil — it's not stamped onto the block but carved out of it as background holes.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="fc7cae8d" lines="9-21">
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

<reference_puzzle index="19" source="train" puzzle_id="b2bc3ffd" lines="8-23">
<description>
Move every object above the solid bottom stripe upward by its own pixel count.

Input structure
The grid has a dominant background, exactly one full-width non-background row
that acts as a fixed stripe, and one or more 4-connected monochrome objects
above that stripe.

Rule
Clear each object from its original cells, then translate it straight upward
by the number of pixels in that object, preserving its shape and color. Leave
the full-width stripe unchanged.

Key insight
Each object's area is also its vertical displacement; the full-width row is a
fixed baseline, not a moving object.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="e4941b18" lines="8-22">
<description>
Move red into teal's top marker position, then move teal to the bottom outside corner of the grey rectangle.

Input structure
- The dominant background surrounds one filled rectangular non-background block.
- There is exactly one red marker and one teal marker, both in the row immediately above the rectangle.
- Both markers are horizontally aligned over columns inside the rectangle.

Rule
- Clear the original red marker.
- Recolor the original teal marker cell red.
- Place the teal marker beside the rectangle's bottom row, on the side indicated by the horizontal direction from red to teal.

Key insight
- The red-to-teal horizontal order chooses the rectangle side where teal exits at the bottom.
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

