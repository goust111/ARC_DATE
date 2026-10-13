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

<reference_puzzle index="2" source="eval" puzzle_id="195c6913" lines="37-52">
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

<reference_puzzle index="3" source="eval" puzzle_id="a6f40cea" lines="74-88">
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

<reference_puzzle index="4" source="eval" puzzle_id="db695cfb" lines="8-19">
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

<reference_puzzle index="5" source="train" puzzle_id="7acdf6d3" lines="25-42">
<description>
Move the source-color pixels into the unique outline interior whose size matches the source count.

Input structure
- The grid has a dominant background and exactly two foreground colors.
- One foreground color forms 8-connected outline objects with row-wise interior cells.
- The other foreground color supplies a set of source pixels away from the chosen outline.
Rule
- For each possible outline color, compute each 8-connected outline object's
  row-span interior: background pixels strictly between the leftmost and
  rightmost outline pixels on each occupied row.
- Select the unique outline interior whose cell count equals the total
  number of pixels of the other foreground color.
- Erase the source pixels and paint that matched interior with the source color.
Key insight
- The colors are roles, not fixed IDs: the source color is identified by
  counting, and the destination is the outline with a matching interior area.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="342dd610" lines="18-36">
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

<reference_puzzle index="7" source="train" puzzle_id="df9fd884" lines="33-47">
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

<reference_puzzle index="8" source="train" puzzle_id="1478ab18" lines="35-43">
<description>
Paint the teal outline of the right triangle determined by the missing corner of the grey-marker square.

Input structure
The grid is an orange background with exactly four foreground marker pixels of one color. The markers have a square bounding box: three markers are bbox corners, and the fourth marker is inside that square.
Rule
Find the one corner of the marker bounding square that has no marker. That missing corner is the right-angle vertex of a right isosceles triangle; paint its two axis-aligned legs and the opposite diagonal in teal, leaving existing marker pixels unchanged.
Key insight
The present corner marker opposite the missing corner is a decoy; the three triangle vertices are the missing corner and the two marker corners adjacent to it.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="6d1d5c90" lines="8-23">
<description>
Remove the left control column and roll the remaining image downward by the red marker's row.

Input structure
- The input is a rectangular grid with a one-cell-wide control column on the left.
- That control column is all pink except for exactly one red marker pixel.
- The columns to the right are the image that remains in the output.

Rule
- Delete the left control column.
- Let r be the row index of the red marker in that column.
- Cyclically shift the remaining image downward by r rows.

Key insight
- The red marker's vertical position encodes the row-roll amount; the pink
  column is only a control strip and is removed after reading it.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="eval" puzzle_id="6e4f6532" lines="26-41">
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

<reference_puzzle index="11" source="train" puzzle_id="2a28add5" lines="8-22">
<description>
Paint each pink control pixel as a teal horizontal bar whose reach equals the foreground counts on that row.

Input structure
- The grid has a dominant background color and one or more pink control pixels.
- Each marker row has exactly one pink pixel; all other non-background pixels
  in that row are counters.
Rule
- Start from an all-background output grid.
- For each pink pixel, count non-background pixels strictly to its left and
  strictly to its right in the same row.
- Paint teal from c - left_count through c + right_count on that row.
Key insight
- The foreground pixels are not copied; only their counts on each side of
  each pink marker encode the teal bar length.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="3cd86f4f" lines="6-18">
<description>
Shear the rectangular input down-left onto a black canvas.

Input structure
A rectangular grid of colored pixels with no internal empty region to infer.

Rule
Create a black output with the same height and width W + H - 1, then copy
each input row r unchanged starting at column H - 1 - r.

Key insight
The colors and row contents are preserved; only the row offsets change so
the left edge forms a descending diagonal.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="c6141b15" lines="84-95">
<description>
Swap a connector segment with repeated marker sprites by connecting the marker anchors and stamping the sprite at the segment endpoints.

Input structure
The grid has one background color and exactly two foreground colors. One foreground color forms a single straight 8-connected segment. The other foreground color forms two or more identical centered sprites.

Rule
Build a background canvas of the same size. Draw straight connector-color segments between every pair of original sprite anchors. Then stamp the marker sprite, using its centered anchor, onto the two endpoints of the original connector segment.

Key insight
The original segment endpoints are the destinations for the repeated sprite, while the original sprite centers become the vertices of the new connector-color complete graph.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="689c358e" lines="72-88">
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

<reference_puzzle index="15" source="train" puzzle_id="ac0c2ac3" lines="7-19">
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

<reference_puzzle index="16" source="train" puzzle_id="db615bd4" lines="111-130">
<description>
Thicken the sparse frame and stamp the external sparse sprites into it in positional order.

Input structure
The grid has a dominant background color, a regular lattice color, one large
sparse rectangular frame, and one or more external monochrome sprites whose
pixels also sit on a two-step lattice.

Rule
Use the largest sparse rectangular perimeter as the frame. Clear that frame
region and the external sprite pixels to background, draw a solid one-pixel
border in the frame color, solidify each sparse sprite into its contiguous
two-step mask, and place the sprites inside the frame with equal gaps.
Wide frames receive sprites left-to-right by source position; tall frames
receive them top-to-bottom by source position.

Key insight
The frame supplies ordered slots: the sprite-to-slot assignment is by visible
source order, not by numeric color or by matching individual cells.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="878187ab" lines="10-24">
<description>
Replace the scattered foreground markers with a bottom-left count rectangle whose two corner diagonals encode the marker counts.

Input structure
- A background grid with exactly two foreground colors used as scattered marker pixels.
- The two foreground colors have distinct pixel counts.

Rule
- Ignore marker positions and use only the two foreground counts.
- Make a 16-by-16 background canvas, fill a bottom-left red rectangle whose height is the smaller count and width is the larger count.
- Paint yellow pixels along the two rising 45-degree diagonals from the rectangle's bottom-left and bottom-right corners.

Key insight
- The marker colors and positions vary between examples; their counts alone determine the output rectangle dimensions.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="d749d46f" lines="32-51">
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

<reference_puzzle index="19" source="train" puzzle_id="5587a8d0" lines="11-27">
<description>
Redraw foreground objects as nested square rings ordered by object size.

Input structure:
- The grid has one dominant background color.
- Every foreground object is a 4-connected monochromatic component.
- Foreground object areas are distinct.

Rule:
- Sort the foreground objects by decreasing pixel count.
- Create a square output with side length 2 * number_of_objects - 1.
- The largest object color fills the outer ring, each next object color fills
  the next inner ring, and the smallest object color fills the center pixel.

Key insight: The objects' original shapes do not get copied; their areas
determine only the outside-to-inside order of the nested square pattern.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="db7260a4" lines="37-56">
<description>
Drop the singleton marker into the matching red U-shaped container, or make a bottom-row puddle if no closed container is hit.

Input structure
- The grid has a dominant background, one singleton marker pixel on the top row,
  and one other foreground color forming vertical bars and bottom segments.
- A closed container is a U shape: two vertical side bars with a continuous
  bottom segment between them and an empty rectangular interior above the bottom.

Rule
- Remove the top marker from its original cell.
- If exactly one closed U-shaped container has the marker column within its
  horizontal footprint, fill that container's empty interior with the marker color.
- If no closed container is under the marker column, fill the whole bottom row
  with the marker color.

Key insight
- The marker does not identify a color by value; it identifies a vertical drop
  column, and only a structurally closed U shape can catch and contain the fill.
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
   - **不建议使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算 / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数等复杂且不易被看出来的计算。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息（例如颜色id是不可用的）或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
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
  网格尺寸合理（≤30×30），使用颜色不超过 10 种，规则复杂度适中，人类能从 3-5 个样例中发现规律。
