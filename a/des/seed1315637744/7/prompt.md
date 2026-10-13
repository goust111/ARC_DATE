你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="8e301a54" lines="8-23">
<description>
Move every foreground object straight downward by a number of rows equal to its own pixel count.

Input structure:
- The grid has one dominant background color.
- The foreground is made of separated 4-connected monochromatic objects.
- Objects may move partly below the bottom edge of the grid.

Rule:
- Clear the canvas to the background.
- For each foreground object, compute its area in pixels.
- Redraw the object at the same columns and that many rows lower, clipping pixels below the grid.

Key insight:
- The downward displacement is the object's own area; color and shape only identify which pixels move together.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="eval" puzzle_id="6e4f6532" lines="26-41">
<description>
Dock train sprites onto station groups by matching coupler count, then orient via divider-color markers.

Input structure:
- bg; colored stripes (edge + internal) partition grid into rectangular regions.
- Color 8 forms train bodies; color 9 marks both isolated stations and train-attached couplers.
- Stripe-colored pixels on trains are orientation markers.

Rule:
- Stations = 8-connected 9-clusters not adjacent to any 8. Trains = components containing 8.
- Match train to station group where |couplers| == |station size|.
- Rotate/flip train so marker pixels face the same-color boundary stripe of the target region.
- Translate so couplers align exactly onto station cells; erase original position.

Key insight: Coupler count selects the docking target; marker color encodes which boundary stripe to face, fixing the unique orientation.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="8abad3cf" lines="10-25">
<description>
Repack each non-background color group as a solid square ordered by increasing size.

Input structure
The grid has one dominant background color, and each other color has a
perfect-square pixel count. Those square side lengths are unique.

Rule
Convert every non-background color into a filled square whose side length is
the square root of its pixel count. Place the squares from left to right in
increasing side length, bottom-aligned, with one background column between
neighboring squares.

Key insight
The original geometry of each color is discarded; only the color's total
pixel count determines the size of its packed output square.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="34cfa167" lines="94-117">
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

<reference_puzzle index="5" source="eval" puzzle_id="271d71e2" lines="9-22">
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

<reference_puzzle index="6" source="train" puzzle_id="a1aa0c1e" lines="74-97">
<description>
Compress the stacked colored bars into a compact ordered table with the control pixel on the selected row.

Input structure
- A background grid contains several full-width colored bars from top to
  bottom; the last bar is the base/separator color.
- Each non-base bar may have an aligned three-column comb motif below it,
  alternating side-pair rows with solid rows.
- A singleton control pixel sits away from the colored motifs.

Rule
- Make one output row per non-base bar, preserving top-to-bottom order.
- In that row, paint the bar color once for each solid three-pixel row in
  its motif, then paint the base-bar color in the separator column.
- The singleton control pixel sits immediately above the base bar. Among
  motifs whose open side-pair parity is compatible with that bottom row,
  put the control pixel in the output row with the fewest solid rows; use
  horizontal distance to the motif center only to break count ties.

Key insight
- The vertical comb height is a count, while the control pixel is read by
  projecting it onto the center gap of compatible comb rows rather than by
  its vertical band alone.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="e048c9ed" lines="11-26">
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

<reference_puzzle index="8" source="eval" puzzle_id="31f7f899" lines="7-21">
<description>
Sort vertical bars by height and redistribute them across fixed colored slots.

Input structure:
- bg background; a horizontal base line of a single color spans the full grid width.
- Several monochrome vertical bars of distinct colors rise upward from the base line.

Rule:
- Rank bar shapes by height (shortest to tallest).
- Rank original x-positions (slots) left to right.
- Place the i-th shortest shape into the i-th leftmost slot, recoloring it to that slot's original color.
- The base line is preserved unchanged.

Key insight: Each slot keeps its x-position and color identity; only the bar silhouettes are redistributed among slots by height order.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="df9fd884" lines="33-47">
<description>
Move the colored object from the lower bracket lane to the upper bracket lane and drop it to the bottom edge.

Input structure
- The background is the dominant color.
- One frame color forms exactly two 4-connected bracket objects: a top-edge bracket open downward and a bottom-edge bracket open upward.
- One other 4-connected foreground object sits in the lower bracket's interior column span.

Rule
- Remove the movable object from its original cells.
- Keep its shape and color unchanged.
- Paste it with its bbox columns aligned to the top bracket's interior columns and with its bbox bottom aligned to the grid bottom.

Key insight
- The lower bracket identifies the object's source lane; the upper bracket provides the target lane, and the final vertical placement is bottom-aligned.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="2f767503" lines="66-81">
<description>
Erase each target-color object crossed by the ray pointed by the bar-and-control marker.

Input structure:
- The background is the dominant color.
- There is one three-pixel straight marker bar and one singleton control pixel adjacent to the bar center.
- The remaining non-background color forms multiple 4-connected target objects.

Rule:
- Cast a straight ray from the marker bar center in the direction away from the adjacent control pixel.
- Find every 4-connected target-color object that intersects that ray.
- Recolor all pixels of those hit target objects to the background; all other pixels stay unchanged.

Key insight:
- The whole object is erased when the center ray touches any of its pixels; the full marker bar width is not swept.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="ad38a9d0" lines="23-36">
<description>
Recolor each placeholder object according to its normalized shape while leaving the background fixed.

Input structure
A same-size grid with one dominant background color and one foreground
placeholder color forming separated 4-connected objects.

Rule
Replace every placeholder object by the color assigned to its exact
translation-normalized mask; all background pixels remain unchanged.

Key insight
The input objects are a shape legend with repeated instances: identical
masks always receive identical colors, independent of their position.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="35ab12c3" lines="73-88">
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

<reference_puzzle index="13" source="eval" puzzle_id="195c6913" lines="37-52">
<description>
Simulate particles along a path with alternating turn direction at collisions.

Input structure:
- Top-left: 2x2 monochrome blocks define a repeating color cycle (left-to-right).
- Isolated single pixels of a cycle color: start markers adjacent to the path.
- A 2x2 monochrome block of a distinct color: end marker.

Rule:
- Each start marker launches a particle moving straight, painting cycle colors in order.
- On hitting a boundary, paint that cell with the end color and turn 90°.
- Turn direction alternates: odd collisions → left (CCW), even → right (CW).
- If the scheduled turn is blocked, the particle stops.

Key insight: The alternating left/right turn rule is the core mechanism; without it the path would be trivial.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="1a244afd" lines="49-68">
<description>
Rotate each magenta source pixel 90 degrees counterclockwise around its matched blue pivot, recoloring the new pixel orange.

Input structure:
- The background is the dominant color.
- The only foreground colors are blue pivot pixels and magenta source pixels.
- Each magenta source can be paired one-to-one with a blue pivot in the same
  row or column.

Rule:
- Remove all magenta source pixels.
- For each matched source/pivot pair, preserve the source distance from the
  pivot but rotate its offset 90 degrees counterclockwise.
- Paint the rotated in-bounds target orange; if the target is outside the
  grid, the source simply disappears.

Key insight:
The blue pixels are rotation centers, and row/column alignment gives a
unique global matching from each magenta source to its pivot.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="142ca369" lines="6-19">
<description>
Trace diagonal rays from triangular arrows, reflecting off colored wall pixels.

Input structure:
- bg grid with monochrome 3-pixel L-shaped "arrows" (2x2 bbox missing one corner) and scattered colored wall pixels.

Rule:
- Each arrow fires a ray diagonally away from its missing corner, painting bg cells in its color.
- A head-on wall hit reverses both axes; a side-touch wall hit flips the touched axis only.
- On any bounce the ray adopts the wall's color and recolors the contact cell.
- Ray stops when it exits the grid or reaches another arrow.

Key insight: Two distinct reflection types (corner vs side-touch) plus color inheritance at each bounce drive the path geometry.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="1818057f" lines="6-19">
<description>
Detect thin '+' shapes in a single foreground color and recolor them to teal.

Input structure:
- bg = most common color.
- One foreground color (e.g. yellow) forms various shapes on the bg, some of which are small '+' crosses.

Rule:
- A '+' center is any foreground pixel whose 4 orthogonal neighbors are also foreground.
- Recolor each such center and its 4 neighbors to teal (8).
- All other pixels remain unchanged.

Key insight: Only compact 5-cell crosses qualify — a pixel embedded in a larger filled region also satisfies the neighbor check, so the rule highlights every local '+' motif, not just isolated ones.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="eval" puzzle_id="cbebaa4b" lines="51-65">
<description>
Assemble scattered jigsaw sprites around a central yellow anchor by pairing red connector pixels.

Input structure:
- bg = black (0). Several colored sprites scattered on the grid.
- One anchor sprite contains a 3×3 yellow (4) block.
- Each sprite has red (2) connector pixels along its edges.

Rule:
- The anchor stays in place; remaining sprites attach one at a time.
- Each sprite is translated so that one of its red pixels overlaps an unpaired red pixel on the growing assembly.
- Every red connector must pair with exactly one other, forming a complete joined structure.

Key insight: Red pixels act as jigsaw pegs — each must match exactly one counterpart, and the assembly order is determined by which unpaired connectors remain.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="41ace6b5" lines="7-26">
<description>
Normalize vertical two-color bars between paired marker columns around the marker rows.

Input structure
- The background is the dominant color.
- A row of red marker pixels sits directly above a row of gray marker pixels
  in the same alternating columns.
- The columns between adjacent markers contain vertical bars made from a
  contiguous teal run above a contiguous blue run.

Rule
- Keep the marker pixels and all non-bar background cells fixed.
- Let h be the maximum teal-run height among the bar columns; paint teal in
  every bar column for h rows ending at the red marker row.
- In each bar column, move its blue run to start at the gray marker row and
  fill all lower cells in that column with maroon.

Key insight
- The marker rows are a baseline: teal is normalized into a common upper
  band, while each blue run keeps its own height below the baseline.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="eval" puzzle_id="5dbc8537" lines="10-23">
<description>
Fill rectangular cavities in a template shape by placing source sprites into matching holes.

Input structure:
- bg = most common color. A dominant "base" color forms a large frame/template with rectangular gaps.
- Small colored sprites sit outside the template region.

Rule:
- The base color (largest connected component excluding bg) defines the template bbox.
- Each rectangular hole inside the template matches exactly one outside sprite by size.
- Place each sprite into its matching cavity; output is the filled template bbox.

Key insight: Sprite bounding boxes match cavity dimensions exactly — it's a jigsaw-fit puzzle where each piece has a unique slot.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="8886d717" lines="43-63">
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

