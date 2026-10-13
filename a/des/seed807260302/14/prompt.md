你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="afe3afe9" lines="106-119">
<description>
Convert each hollow 3x3 symbol to one pixel and slide the pixels clockwise from the marked edge.

Input structure
- The grid has a background, one full-edge marker line, and monochrome hollow 3x3 foreground symbols.
- The symbols lie on a coarse row/column lattice; fully empty lattice rows or columns are absent from the output.

Rule
- Replace each hollow 3x3 symbol with one pixel of its color in the corresponding coarse-grid cell.
- The marker edge gives the compaction direction by clockwise border travel: top -> right, right -> down, bottom -> left, left -> up.
- Preserve the order of non-background pixels while sliding them as far as possible in that direction.

Key insight
- The large picture is a scaled symbolic grid, and the blue edge marker acts as a gravity direction rather than as output content.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="eval" puzzle_id="142ca369" lines="6-19">
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

<reference_puzzle index="3" source="train" puzzle_id="4a1cacc2" lines="8-23">
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

<reference_puzzle index="4" source="train" puzzle_id="758abdf0" lines="54-69">
<description>
Advance short marker bars away from the full edge bar, turning completed two-pixel markers into edge-color marks at the opposite side.

Input structure
- The background is the dominant color.
- One non-background color forms a complete one-cell-thick bar on exactly one outer edge.
- The other foreground color appears only in the first two cells inward from that bar, lane by lane.

Rule
- In each lane perpendicular to the edge bar, a one-pixel marker adjacent to the bar grows one more cell inward.
- A two-pixel marker in those first two inward cells is cleared back to background.
- The cleared two-pixel marker is replaced by a two-pixel mark of the edge-bar color at the opposite side of the same lane.

Key insight
- The full edge bar defines both the orientation and the direction of the local two-step marker cycle.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="22806e14" lines="23-37">
<description>
Mark the center of each odd-sized solid square with the plus marker color, erasing the marker only when it is outside the marked-center field.

Input structure:
- The dominant color is background.
- Exactly two foreground colors are present: one unique 3x3 plus marker and one color forming separated solid square objects.

Rule:
- For every non-marker 4-connected object, assert it is a solid square.
- If the square side length is odd, recolor its center pixel to the plus marker color; even-sided squares stay unchanged.
- Preserve the plus marker when its center lies inside the bounding box of the odd-square centers; otherwise erase it to background.

Key insight:
- Odd-sided solid squares have a single visible center pixel, and the plus supplies the color used to identify those centers.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="42f14c03" lines="75-83">
<description>
Crop the foreground object whose missing cells match marker shapes, and fill those missing cells with the marker colors.

Input structure
A dominant background surrounds several monochromatic foreground objects. Exactly one object has background gaps inside its bounding box whose component shapes are matched by separate marker objects.
Rule
Find that object, crop its bounding box, and replace each background gap component with the color of the separate marker component or components having the same normalized shape.
Key insight
The marker objects do not mark absolute positions; their shapes identify which gaps in the selected object should be colored, while objects with unmatched gaps are distractors.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="84551f4c" lines="42-53">
<description>
Rotate each blue-started chain of bottom-aligned vertical bars into horizontal bars resting on the same baseline.
Input structure
- The grid has a background and bottom-aligned, one-cell-wide vertical bars.
- Blue bars are markers; red bars are body bars that may be chained after a marker.
Rule
- Starting at each blue bar, repeatedly include the next red bar whose column is exactly the current bar's column plus the current bar's height.
- Clear every included vertical bar, then redraw each included bar horizontally to the right from its original bottom cell, with length equal to its vertical height.
- Red bars that are not reached from a blue marker remain unchanged.
Key insight
- A vertical bar's height is also the width it occupies after rotation, so the chain follows immediate horizontal contact after each rotated bar.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="320afe60" lines="41-56">
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

<reference_puzzle index="9" source="train" puzzle_id="e39e9282" lines="73-81">
<description>
Project maroon markers into adjacent grey or pink 3x3 squares, erasing grey squares afterward.

Input structure
The grid has a dominant background, solid 3x3 grey and/or pink squares, and maroon marker pixels or segments directly touching one side of exactly one square.
Rule
Grey squares are erased; their adjacent maroon markers stay in place and stamp the nearest edge cell inside the grey square. Pink squares stay pink; their adjacent maroon markers disappear and stamp the corresponding cell on the square's middle row or middle column.
Key insight
The side touched by a marker determines the projection: grey uses the near edge, while pink uses the centerline, preserving the marker's coordinate along that side.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="1efba499" lines="70-85">
<description>
Swap paired marker pixels across the largest foreground-color barrier, placing them just outside the barrier segment.

Input structure:
- The background is the dominant color.
- One foreground color is the unique most frequent non-background color; its pixels form the stationary barrier.
- Other foreground pixels are single markers, sometimes paired on opposite sides of one contiguous barrier segment in the same row or column.

Rule:
- Scan each row and column that crosses exactly one contiguous barrier segment.
- If the line has exactly one marker on each side of that segment, erase both source markers and place each marker in the cell immediately beyond the opposite side of the segment.
- Lines with markers on only one side are unchanged; the barrier remains unchanged.

Key insight:
- The marker pairing is local to a row or column, and the destination is defined by the nearest cells outside the intervening barrier segment, not by the original distance from the barrier.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="b7256dcd" lines="18-32">
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

<reference_puzzle index="12" source="train" puzzle_id="1d61978c" lines="8-22">
<description>
Recolor the two diagonal foreground-bar orientations, assigning teal to the larger orientation and red to the smaller one.

Input structure
- The grid has one dominant background color and one foreground color.
- Foreground pixels form one-cell-thick diagonal bars.
- Every foreground pixel belongs to exactly one diagonal orientation.

Rule
- Keep the background unchanged.
- Classify each foreground pixel by whether it has a diagonal neighbor on the NW/SE axis or the NE/SW axis.
- Recolor the orientation with more foreground pixels teal and the other orientation red.

Key insight
- The output colors depend on the relative population of the two diagonal orientations, not on a fixed slash direction or position.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="15660dd6" lines="33-54">
<description>
Combine each column's repeated template mask with its single solid color swatch.

Input structure:
- The grid is a table of framed panels separated by background rows/columns.
- The leftmost column gives one solid row-label color for each panel row.
- In each panel column, exactly one panel has a solid non-background interior,
  while the other panels share the same mask drawn in one template color on
  the background.

Rule:
- For every panel column, find the row containing the solid interior swatch.
- Rebuild one output panel using that row's left label as the frame color.
- Fill the repeated template mask with the swatch color and leave the mask
  background cells as background; separate rebuilt panels with one
  background column.

Key insight:
The table encodes a colorized stencil per column: the solid panel supplies
the paint color and frame label, while the duplicate template panels supply
the shape mask.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="a09f6c25" lines="33-50">
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

<reference_puzzle index="15" source="eval" puzzle_id="e87109e9" lines="68-83">
<description>
Emit four teal beams from a square, turning at colored obstacles per a legend.

Input structure:
- Top band: grey (5) legend with colored vertical bars in cells; bar position (left or right) encodes turn direction.
- Below: main grid with bg, a solid teal (8) square, and colored obstacle blocks.

Rule:
- Remove the legend from the output.
- From the teal square, shoot a beam in each cardinal direction (as a sliding copy of the square).
- Each beam paints teal over bg cells as it moves.
- On hitting an obstacle of color C, consult the legend: bar on left → turn counterclockwise, bar on right → turn clockwise. Continue in the new direction.
- Beams stop at grid edges.

Key insight: The legend's bar position (left vs right within its cell) is the encoding for turn direction at each obstacle color.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="train" puzzle_id="58c02a16" lines="10-26">
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

<reference_puzzle index="17" source="train" puzzle_id="25c199f5" lines="30-44">
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

<reference_puzzle index="18" source="train" puzzle_id="9b30e358" lines="9-23">
<description>
Repeat the bottom foreground motif upward until it fills the whole grid.

Input structure
- The background is the dominant color.
- The only foreground pixels occupy a contiguous block of rows at the bottom.
- That bottom block is the complete vertical motif, including its background cells.

Rule
- Copy the bottom motif rows repeatedly upward across the full grid.
- Preserve horizontal alignment and vertical phase: the original first motif row remains at its input row.
- The output shape is unchanged.

Key insight
The blank rows above the bottom foreground block are missing repetitions of the same row motif, not empty space.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="4e7e0eb9" lines="31-49">
<description>
Transform each 9x9 panel by mirroring across its center guide line or filling all four block slots from its lone exemplar.

Input structure
The grid is made of 9x9 panels separated by one-cell-wide grid lines. Each
panel has four monochrome 3x3 slots at the quadrant positions, with either a
full non-background center row, a full non-background center column, or no
guide line.

Rule
A panel with a full center column is reflected left-to-right, and a panel
with a full center row is reflected top-to-bottom. A panel with no guide line
has exactly one slot color that appears once and one slot color that appears
three times; recolor all four slots to the once-occurring slot color.

Key insight
The center guide line encodes the mirror axis, while panels without a guide
line are completed by copying the structurally unique slot color.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="train" puzzle_id="52df9849" lines="49-67">
<description>
Complete each partially hidden simple foreground shape by recoloring the occluding pixels in its footprint.

Input structure
- The grid has a dominant background and several solid foreground shapes.
- Some shapes are partially hidden by other foreground colors, but their full
  footprint is still indicated by a foreground-filled bbox or by a foreground-
  filled 45-degree diagonal line.

Rule
- For each foreground color, complete its full rectangular bbox when that bbox
  contains no background and includes other colors.
- Otherwise, complete its 45-degree diagonal line when all visible pixels lie
  on that line and the hidden line cells are foreground, not background.
- Recolor only those hidden footprint pixels; leave all other pixels unchanged.

Key insight
- The occluded cells are exactly the non-background cells inside a simple
  shape's implied footprint, so completing shapes is a local recoloring problem.
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

