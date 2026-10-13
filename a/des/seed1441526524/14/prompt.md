你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="dc46ea44" lines="11-25">
<description>
Move the lower sprite and its pink guide into the upper panel, aligning the sprite to the guide's lower-left branch.

Input structure
- The grid has one dominant background color and one full-width separator row.
- Below the separator are exactly two foreground roles: a five-pixel guide marker and one monochrome sprite.
- The guide marker has normalized shape {(0,2), (1,2), (2,0), (2,2), (3,1)} and starts immediately below the separator.

Rule
- Clear the canvas to the background, preserving the separator row.
- Translate the guide marker upward so its top row becomes row 0.
- Move the sprite, preserving its color and shape, so the sprite bbox bottom-right aligns to the translated guide's lower-left branch.

Key insight
- The marker supplies the destination anchor: its lower-left branch cell is where the moved sprite's bbox bottom-right must land.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="66ac4c3b" lines="57-72">
<description>
Complete the sparse-color reflected copy of the foreground motif across its matching marker row or column.

Input structure
- The grid has one background color and two foreground colors.
- One foreground color appears only as a sparse marker row or column.
- The other foreground color contains the same sparse marker positions on a parallel row or column, plus the motif on the far side of the sparse markers.

Rule
- Keep the original grid unchanged.
- Reflect every motif pixel of the larger foreground color across the two parallel marker lines.
- Paint the reflected motif pixels with the sparse marker color.

Key insight
- The two equal marker rows or columns are paired landmarks: the sparse-color marker line borders the source motif, and the matching source-color marker line anchors the reflected copy.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="train" puzzle_id="b25e450b" lines="28-43">
<description>
Slide each black border rectangle to the opposite border, recoloring its swept stripe orange and leaving the rectangle black at the far edge.

Input structure
- The grid contains black rectangular 4-connected masks, each touching exactly one outer border.
- Grey cells form incidental pattern content; orange is the trail color used when a black mask slides.

Rule
- Treat all black rectangles as moving simultaneously.
- A rectangle touching the top or bottom edge moves vertically to the opposite edge in the same columns.
- A rectangle touching the left or right edge moves horizontally to the opposite edge in the same rows.
- Recolor the complete swept row or column stripe to orange, then stamp every destination rectangle in black.

Key insight
- Edge contact gives the slide direction; the output is the set of opposite-edge black rectangles plus their orange trails.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="2ba387bc" lines="6-21">
<description>
Sort scattered 4×4 squares into two columns by type: frames left, solids right.

Input structure:
- bg = most common color. Multiple 4×4 objects on bg.
- Frames: colored 1px border, bg-filled 2×2 interior.
- Solids: entirely filled with one non-bg color.

Rule:
- Classify each 4×4 object as frame or solid by its interior.
- Sort each group top-to-bottom by input row.
- Stack frames in left column (cols 0–3), solids in right (cols 4–7).
- Output is 8 wide; height = 4 × max(#frames, #solids), bg-padded.

Key insight: Frame vs solid is determined solely by whether the 2×2 interior matches bg.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="5ad8a7c0" lines="10-24">
<description>
Connect the innermost mirrored marker pair(s) by filling their horizontal span.

Input structure
- The grid has one background color and one foreground color.
- Each foreground row contains exactly two foreground pixels mirrored across the vertical center.

Rule
- Find the foreground row pair(s) with the smallest horizontal gap between the two markers.
- Fill every pixel between those two markers in each selected row with the foreground color.
- Leave all other pixels unchanged.

Key insight
- Wider marker pairs are context; the closest marker pair(s) show the unresolved horizontal segment.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="c61be7dc" lines="42-57">
<description>
Flatten the interior object into a centered line and move the guide bars inward to bracket it.

Input structure:
- The dominant color is the background.
- One non-background guide color touches the boundary and forms exactly two full parallel bars plus a split perpendicular center bar.
- The other non-background color is a single 8-connected interior object whose area determines the flattened line length.

Rule:
- If the guide pair is vertical, make a vertical object-colored line centered between those bars; otherwise use the analogous horizontal line.
- The line length equals the number of interior object pixels and is centered on the split perpendicular guide bar.
- Rebuild the guide color as the perpendicular center bar plus two full bars immediately beside the flattened object line.

Key insight:
- The guide bars supply only the orientation and center; the interior object's pixel count supplies the length of the final one-cell-thick line.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="e734a0e8" lines="44-59">
<description>
Copy the unique colored sprite tile into every tiled cell marked by an internal separator-colored pixel.

Input structure
- Full same-color separator rows and columns partition the grid into equal rectangular tiles.
- One tile contains the only non-background, non-separator colored sprite.
- Target tiles are otherwise background-colored but contain at least one internal pixel of the separator color.

Rule
- Keep the separator lines and the source sprite tile unchanged.
- Replace each marked target tile with an exact copy of the source tile.
- Unmarked background-only tiles remain unchanged.

Key insight
- The internal separator-colored pixels are placement markers; the single colored tile is the template to stamp into those marked cells.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="d6e50e54" lines="8-25">
<description>
Move each maroon marker toward the blue rectangle and recolor the rectangle red.

Input structure
A background canvas contains one solid blue rectangular object and several isolated
maroon marker pixels. Each marker is horizontally or vertically aligned with the
rectangle and lies outside it.

Rule
Start with a background canvas, repaint the blue rectangle red, and move each
maroon marker straight toward the rectangle along its row or column. Exactly one
marker is strictly closer to the rectangle than all others and enters the
rectangle boundary; every other marker lands in the adjacent exterior cell just
outside its aligned side.

Key insight
The single entering marker is the unique global nearest marker to the rectangle,
not the nearest marker on each of the four sides.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="eval" puzzle_id="6e4f6532" lines="26-41">
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

<reference_puzzle index="10" source="train" puzzle_id="252143c9" lines="14-33">
<description>
Keep only the center color and redraw it as a diagonal ray to its indicated corner.

Input structure:
- The grid is an odd square with a unique dominant background color.
- The center pixel is foreground.
- Every off-center pixel with the center color lies in the same diagonal
  quadrant relative to the center.

Rule:
- Fill the whole output with the background color.
- Use the quadrant of the center-colored pixels to choose one diagonal
  direction from the center.
- Draw a continuous line of the center color from the center to the grid
  edge in that direction.

Key insight:
The color at the exact center identifies the surviving color, while the
other pixels of that color identify which corner the center should connect to.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="train" puzzle_id="9b30e358" lines="9-23">
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

<reference_puzzle index="12" source="train" puzzle_id="ac0c2ac3" lines="7-19">
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

<reference_puzzle index="13" source="train" puzzle_id="1b59e163" lines="54-69">
<description>
Copy each marked sprite to every standalone marker of the same color and erase the original sources.

Input structure:
- The background is the dominant color.
- Each source is an 8-connected component containing a monochrome sprite and one marker pixel of another color.
- Other isolated marker pixels may be destinations; marker colors with no source sprite are distractors.

Rule:
- Clear the grid to background.
- For every source marker color, place the sprite at each other marker of that color using the source marker as the anchor.
- Keep each destination marker pixel in place; do not keep source sprites, source markers, or distractor markers.

Key insight:
- A colored marker is not part of the sprite geometry; it is the anchor that says where same-colored copies of that sprite should be stamped.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="train" puzzle_id="e4941b18" lines="8-22">
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

<reference_puzzle index="15" source="train" puzzle_id="79cce52d" lines="29-44">
<description>
Crop off the control row and column, then cyclically roll the payload by the two marker offsets.

Input structure
- The grid is square, with a control row on top and a control column on the left.
- The control row and column share one filler color and one single-pixel marker color.
- The lower-right square is the payload and contains none of the control colors.

Rule
- Remove the top control row and left control column.
- Roll the remaining payload down by the marker's zero-based position in the left control column.
- Roll the payload right by the marker's zero-based position in the top control row.

Key insight
- The two border markers are cyclic shift amounts for the payload, not colors to preserve.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="5dbc8537" lines="10-23">
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

<reference_puzzle index="17" source="train" puzzle_id="6a980be1" lines="46-61">
<description>
Project the framed periodic stripe pattern across the canvas and erase the unused frame cells.

Input structure
- A bg canvas with a one-cell-thick rectangular frame of one color.
- Inside the frame are two parallel full-interior marker bars of another color.
- A third color fills the region between the marker bars on periodically spaced rows or columns.

Rule
- Infer the periodic row or column mask from the third-color stripes inside the frame.
- Paint the frame color along those full rows or columns, extended to the grid boundary.
- Extend the marker bars across the full canvas and repaint the source-color stripe cells between them.

Key insight
- The outer frame is not preserved as a border; it supplies the color for the periodic stripes selected by the inner source pattern.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="train" puzzle_id="5e6bbc0b" lines="42-56">
<description>
Compact the checkerboard's moving color toward the edge marked by the teal pixel, leaving a maroon swept trail on the marker line.

Input structure
- A blue/black checkerboard with exactly one teal marker replacing a blue pixel on a non-corner edge.
- The marker's checkerboard parity identifies the moving color (blue); the opposite parity is background (black).

Rule
- If the marker is on the left or right edge, compact moving-color pixels in every row toward that side.
- If the marker is on the top or bottom edge, compact moving-color pixels in every column toward that side.
- Preserve the teal marker at its edge cell.
- On the marker's row or column, color cells swept by shifted moving pixels maroon unless the final compacted line occupies them.

Key insight
The marker selects both the compression direction and the checkerboard color class that moves; maroon records the movement trail only along that marked line.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="4a1cacc2" lines="8-23">
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

<reference_puzzle index="20" source="train" puzzle_id="1be83260" lines="41-58">
<description>
Decode the colored slot map inside the marked motif and rebuild the motif lattice with those colors.

Input structure:
- A blank background surrounds a rectangular lattice of identical motifs.
- The motif rows and columns are separated by one background line.
- One motif has colored marker pixels in its internal odd-row/odd-column slots.

Rule:
- The motif size is 2R+1 by 2C+1, where R by C is the motif lattice size.
- Read the marked motif's odd internal slots as the color map for the R by C output motifs.
- Slots that still contain the repeated motif color map to that repeated motif color.
- Rebuild the compact lattice on a canvas filled with the repeated motif color,
  painting each motif mask with its decoded color.

Key insight:
- The holes and filled odd slots inside one motif are a miniature legend for recoloring the entire repeated motif lattice.
</description>
</reference_puzzle>

你的任务：
以参考题为标杆，运用发散性思维进行创新扩展。请发挥创造力，构思全新且巧妙的推理谜题。用中文创造全新的 ARC 谜题描述。
创新方式包括但不限于：
**机制重组**：将多道参考题各自的核心机制提取出来，组合成一个新的多步管道，用到新的场景。
**机制变异**：取一道参考题的核心机制，以机制为基础进行创新并与其他多种机制进行创新重组，使解题路径发生质变而非量变。
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
   - **允许使用**（这些都是人类能心算的视觉量）：数像素 / 孔洞 / 对象 / 邻居（≤10 量级）、比大小、按某个可见属性排序、奇偶判断、取最大 / 最小、用一个计数值作为另一个等量几何量（如 N 个像素 → 长度 N 的射线、面积 N → 边长 N）、判断接触 / 相邻 / 包含、粗略中心位置（簇的视觉中点）、"距离"作为"几步"而不是公式。
   - **不建议使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算 / 余数、拓扑距离（BFS / 最短路 / 跳数）、高精度方向区分（>4 方向，如时钟 12 方向）、精确几何质心坐标、凸角 / 凹角计数。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
4. 生成的问题描述旨在促进后续用于创建题目网格的代码生成。因此，对输入 / 输出网格的描述以及规则总结必须极其详细；语言风格应易于理解；对于特定对象（如箭头），必须在描述中明确其绘制方式。
5. 描述谜题时避免使用 0 或 1 这类颜色编号；应使用颜色名称，例如 "red"。可用颜色如下：Black、Blue、Red、Green、Yellow、Grey、Magenta、Orange、Azure、Maroon。
6. 每个新题的核心推理机制应明显不同——即从观察样例到归纳规则的认知路径不同，而不仅仅是场景、对象或颜色不同。
7. 题目难度应来自机制的隐蔽性而非步骤数。用参考题作为难度标尺——如果你生成的题一个样例就能猜到规则，说明机制太透明，需要加深隐蔽性。
8. 优先从参考题的机制出发进行重组或推演，而非从物理隐喻出发。物理隐喻容易导致大量题目归结为一种模式，例如"沿方向移动到阻挡处停止"。
9. 题目的难度优先来自规则的**结构层数**，而不是单一步骤的复杂度。关于"结构层数"的判定标准：**只有解题者需要主动推理才能完成的步骤才算一层**。肉眼一扫即可获得的信息（如"找到缺口在哪一侧"、"数出像素个数"、"识别哪个对象最大"）不构成独立的推理层——它们只是变换的前置输入，不增加认知难度。真正的多层结构是指每一层都需要解题者发现一个非显然的规律。
   如果你发现自己写出的题只是"对每个对象，根据它的某个属性做某变换"的单步规则，请引入一个需要主动推理的前置步骤。**用堆叠 tie-break 规则或复杂数学公式来"加难度"是劣解；用多一层需要归纳的结构嵌套来"加难度"才是好解**。
10. 层数可以是两层、三层或更多——不必拘泥于两层。若题目的机制天然适合更深的嵌套，鼓励向三层及以上尝试；但层数本身不是目的，每一层都必须有独立的归纳价值。

生成后对生成的题执行以下五项审查（审查内容不需要保存到文件）：
1. 原创性
  新题不可与上述参考题相同或类似，不可只是替换颜色、方向、对象类型等表层要素。判断标准：新题的核心推理路径应与任何一道参考题不同。
2. 确定性
  规则完全确定，同一输入只能产生唯一输出，每一步操作都有明确定义。
3. 可实现性
  所有必要细节都已给出——对象如何生成、如何识别、如何变换、边界情况如何处理，读完描述后可以直接写代码。
4. ARC 合规性
  网格尺寸合理（≤30×30），使用颜色不超过 10 种，规则复杂度适中，人类能从 3-5 个样例中发现规律。
5. 层级有效性与内部一致性
  逐题检查以下两点，发现问题需要重写该题：
  - **层级是否会坍缩**：每一层是否都对最终输出有独立贡献？是否存在某一层无论怎么变化都不影响下一层结果的情况？（例如规则号称"先用 A 选出 B，再用 B 控制 C"，但 A 的所有可能取值都导出同一个 B，第一层实际无效，整道题坍缩为单层。）
  - **规则是否自相矛盾**：题目内部是否存在隐性冲突或未覆盖的退化情况？（例如两条优先级条件在某些输入下会冲突，或某个对象没有标记、某个区域无内部像素时规则失效。）
