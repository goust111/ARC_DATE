你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="320afe60" lines="41-56">
<description>
Slide each foreground object horizontally to the edge indicated by whether its shape has an open notch.

Input structure:
- Yellow is the background and there is one blue foreground color.
- The foreground is split into separated 4-connected objects.
- Each object keeps its row span and shape in the output.

Rule:
- If an object's bbox border contains a background gap flanked by object pixels along that same border, slide it to the right edge and recolor it green.
- Otherwise slide the object to the left edge and recolor it red.
- Start from a blank background canvas; no moved objects overlap.

Key insight:
- The target side is encoded by the object's shape, not by its original position: open notches go right, while closed or solid shapes go left.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="9f8de559" lines="19-37">
<description>
Erase the first border pixel hit by extending the red marker away from its pink trail.

Input structure
- The dominant background surrounds a unique red marker and a four-pixel
  pink trail.
- The pink pixels form one straight 8-connected ray starting next to the
  red marker.
- In the opposite direction from that trail, the red marker faces a
  non-background border or obstacle pixel beyond background cells.

Rule
- Keep the whole grid fixed except for the first non-background pixel on
  the ray from the red marker away from the pink trail.
- Recolor that one hit pixel to the background color.

Key insight
- The pink trail marks where the red marker came from, so the edit is found
  by ray-casting from the red marker in the opposite direction.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="5792cb4d" lines="44-59">
<description>
Reverse the sequence of colors along the single 4-connected foreground path while keeping the path shape fixed.

Input structure:
- The background is the dominant color.
- All non-background pixels form one multicolor 4-connected one-pixel-wide path.
- The path has exactly two endpoints and no branches.

Rule:
- Walk the path from one endpoint to the other.
- Replace the colors on those same path cells by the original path colors in reverse order.
- Leave every background pixel unchanged.

Key insight:
- Nothing moves geometrically; only the color string written on the path is reversed between the endpoints.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="269e22fb" lines="46-60">
<description>
Complete a visible crop into the shared 20x20 binary motif in its matching orientation.

Input structure:
- The grid is a two-color rectangular crop of one rotation/reflection of the same 20x20 motif.
- One color is the motif foreground and the other color is the motif background.

Rule:
- Try each input color as the foreground mask.
- Match that mask against every possible window in every dihedral orientation of the motif.
- Repaint the uniquely matched full 20x20 motif with the matched foreground/background colors.

Key insight:
- The examples are partial windows of one canonical binary pattern; the local crop is distinctive enough to identify the full orientation and offset.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="58c02a16" lines="10-26">
<description>
Expand the top-left seed mask into a recursive colored pattern and tile it across the full grid.

Input structure
- The background is the dominant color.
- All foreground lies in a top-left rectangle whose last row and last column are a solid secondary color.
- Removing that secondary-color row and column leaves a seed region containing background plus one primary color.

Rule
- Use the primary-color pixels in the seed as a binary mask.
- Replace each primary seed cell with a copy of the seed mask in the primary color, and each background seed cell
  with a copy of the seed mask in the secondary color.
- Repeat and crop that expanded tile to the original grid shape.

Key insight
The solid L-shaped secondary-color border is a legend, not output content; it supplies the color used when the
seed's background cells are expanded.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="42f83767" lines="23-39">
<description>
Expand the lower color grid by replacing each pixel with its matching top stencil.

Input structure
The input has a top legend and a lower dense color grid, separated by blank rows.
The legend contains left-to-right 2x2 color swatches and, to their right,
the same number of square neutral-color stencil masks.

Rule
Pair the swatches and stencil masks by left-to-right order. For every pixel in
the lower color grid, paste the paired stencil mask as one output block,
recoloring the mask pixels with that pixel's color and leaving mask holes as bg.

Key insight
The lower grid is not copied directly; it is an index grid whose colors select
the stencil shape from the legend, so each input pixel becomes one square block.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="eval" puzzle_id="65b59efc" lines="9-24">
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

<reference_puzzle index="8" source="train" puzzle_id="985ae207" lines="82-97">
<description>
Repeat each 3x3 control-center sprite toward the matching solid rectangle until a copy first overlaps that rectangle.

Input structure
- The background is the dominant color.
- Each sprite is a 3x3 foreground object whose eight border pixels have one color and whose center has a different control color.
- For every control color, there is exactly one separate solid rectangular object of that color.

Rule
- Match each sprite to the unique solid rectangle whose color equals the sprite center.
- Copy the whole 3x3 sprite in the cardinal direction of that rectangle, stepping by three cells each time.
- Stop after pasting the first copy whose 3x3 bbox overlaps the rectangle; copied pixels overwrite the rectangle there.

Key insight
The center pixel is not the paint color: it is a destination key, while the entire 3x3 sprite is the motif tiled along the straight path.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="52364a65" lines="10-24">
<description>
Erase the two leftmost columns of every foreground object, leaving the rest of each object in place.

Input structure:
- A grid has one dominant background color and several monochromatic 4-connected foreground objects.
- Each foreground object spans at least two columns in its bounding box.

Rule:
- Find each same-color 4-connected foreground object.
- Within each object's own bounding box, set foreground pixels in the two leftmost columns to the background color.
- Leave all other pixels unchanged.

Key insight:
- The deletion is measured from each object's local left edge, not from the grid edge or from a shared global column.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="305b1341" lines="22-37">
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

<reference_puzzle index="11" source="train" puzzle_id="a09f6c25" lines="33-50">
<description>
Recolor each non-singleton red object by its symmetry axis and erase isolated red pixels.

Input structure:
- The grid has one dominant background color and red 4-connected foreground objects.
- Every non-singleton red object has exactly one reflection symmetry category:
  horizontal, vertical, or diagonal.
- Single red pixels are noise.

Rule:
- Start from an all-background output of the same shape.
- Recolor horizontally symmetric red objects blue, vertically symmetric objects
  green, and diagonally symmetric objects pink.
- Leave isolated red pixels as background.

Key insight: The target color is not tied to object position or size; it is a
label for the object's reflection symmetry.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="57edb29d" lines="88-103">
<description>
Extract the single plain rectangle and redraw it with the aligned rectangles' border-relative marker pattern.

Input structure:
- The boundary color is the background/separator.
- Foreground objects are dense rectangular regions; exactly one rectangle has only its base color.
- Other rectangles contain their base color plus marker pixels.

Rule:
- Crop the only plain rectangle; this crop is the output canvas.
- Use marked rectangles whose row span overlaps the plain rectangle's row span.
- Transfer their marker pixels by preserving each marker run's distance from the nearest rectangle edges.

Key insight:
- The marked rectangles are examples of an edge-relative pattern, so the missing plain rectangle receives the same pattern resized to its own dimensions.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="230f2e48" lines="19-38">
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

<reference_puzzle index="14" source="train" puzzle_id="25c199f5" lines="30-44">
<description>
Pack the non-background rows from three separated panels into one panel, scanning panels right-to-left.

Input structure:
- A rectangular grid contains three equal-size panels separated by two full-height divider columns.
- The background is the dominant color; divider pixels use a color that appears only in the divider columns.

Rule:
- Read panels from right to left, and within each panel read rows from top to bottom.
- Keep each row that contains any non-background pixel, discarding all-background rows.
- Place the kept rows at the bottom of a single panel-shaped output, padding above with background rows.

Key insight:
- The task is row packing: the divider columns mark independent panels, and only the occupied rows are moved.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="f341894c" lines="63-78">
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

<reference_puzzle index="16" source="eval" puzzle_id="6e453dd6" lines="39-52">
<description>
Slide black shapes rightward to a grey separator, then shoot red rays from enclosed holes.

Input structure:
- A vertical grey (5) separator column divides the grid into left and right halves.
- Black shapes sit on the left side against a background color (bg).

Rule:
- Translate each black connected component rightward until its right edge is flush with the separator.
- After sliding, find enclosed bg regions (holes) within the black shapes that touch the column just left of the separator.
- For each row spanned by such a hole, emit a red ray across the entire right half of the grid.

Key insight: Only holes adjacent to the separator trigger rays — the sliding creates enclosures that act as "nozzles" projecting rightward.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="33067df9" lines="32-48">
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

<reference_puzzle index="18" source="eval" puzzle_id="16b78196" lines="6-20">
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

<reference_puzzle index="19" source="eval" puzzle_id="a6f40cea" lines="74-88">
<description>
Complete partial paths inside a framed rectangle by projecting external color traces inward.

Input structure:
- bg, a 1px rectangular frame (border color) with uniform interior fill.
- Colored components outside the frame touching its edges.

Rule:
- Output = the frame's interior, filled with the fill color.
- Map each outside component's frame-touch points into interior coordinates.
- Monochrome components become solid Manhattan stripes between mapped endpoints.
- Two-color components become alternating-color Manhattan paths.

Key insight: Endpoint colors of two-color traces anchor the alternating phase deterministically.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="20fb2937" lines="63-77">
<description>
Replace each marker below the separator with the mapped top legend square centered on that marker.

Input structure:
- The background surrounds a top legend, one solid non-background separator row, and a lower marker field.
- The legend contains equal-size filled square sprites, with a singleton marker below each sprite and aligned to its center column.
- The lower field contains only background and marker colors from the legend.

Rule:
- Pair legend markers to legend sprites by left-to-right position.
- Discard the legend and separator; keep the grid width and the rows below the separator.
- In the lower field, replace every marker pixel by a filled square of the paired sprite color, centered at the marker location.

Key insight: The top area is a positional color legend, and the actual output is the post-separator marker field expanded by that legend.
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

