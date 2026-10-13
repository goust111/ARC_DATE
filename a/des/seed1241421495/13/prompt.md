你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="89565ca0" lines="120-136">
<description>
Convert each colored partitioned frame into a horizontal bar whose length is its number of enclosed regions.

Input structure:
- Black background with several axis-aligned colored line drawings.
- The line drawings are damaged by scattered pixels of one non-background noise color and by overlaps with other drawings.
- Each real drawing is a rectangular partitioned frame enclosing one or more background regions.

Rule:
- For each non-background color, reconstruct its horizontal and vertical wall segments from dense row/column evidence.
- Count the 4-connected background regions fully enclosed by those reconstructed walls.
- The color with no enclosed region is the noise/pad color.
- Sort drawing colors by enclosed-region count; equal counts are ordered by which reconstructed frame covers the other.

Key insight:
- The scattered color is not an object; it only pads the histogram, while each true object's room count is stable after rectilinear wall reconstruction.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="5b37cb25" lines="59-73">
<description>
Fill each plus-shaped background notch with the color of the frame side it opens toward.

Input structure
- A one-cell-thick rectangular frame has four solid-colored sides and neutral corner pixels.
- The frame interior contains a dominant background color and one foreground object color.
- Some object boundaries contain five-pixel plus-shaped background notches.

Rule
- Find every background plus whose opposite side is supported by foreground object pixels and whose open side faces one frame side.
- Recolor the five pixels of that plus with the color of the facing frame side.
- Leave all other pixels unchanged.

Key insight
The frame side colors are direction labels for the open notches in the foreground object.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="7ed72f31" lines="8-22">
<description>
Mirror colored sprites using red (2) markers as reflection axes.

Input structure:
- bg = most common color. Red (2) = marker color.
- Colored sprites appear near red markers.

Rule:
- Red line within a sprite (vertical or horizontal) → mirror payload across that axis.
- Single red pixel in a mixed sprite → point-reflect payload around it.
- Isolated red pixel (alone, no payload) → point-reflect each adjacent 8-connected color component.
- Originals remain; mirrored pixels overlay.

Key insight: Red marker geometry (line vs single pixel) determines reflection type; isolated vs embedded red determines what gets reflected.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="5ecac7f7" lines="23-36">
<description>
Combine the left, middle, and right ordered objects from three separated panels.

Input structure
- The grid is three HxH panels separated by one-cell vertical separator bars.
- The background is the dominant color, and each panel contains exactly three same-color 4-connected foreground objects.

Rule
- Build a new HxH background canvas.
- From the left panel copy its leftmost object, from the middle panel copy its middle object, and from the right panel copy its rightmost object.

Key insight
- The separator bars create three aligned layers, and the panel's own left-to-right object order selects which object survives in the merged output.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="9473c6fb" lines="10-25">
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

<reference_puzzle index="6" source="train" puzzle_id="11dc524f" lines="25-40">
<description>
Slide the non-square sprite to the square anchor and replace the anchor with the sprite's mirror image.

Input structure:
- The background is the most common color.
- There is one 8-connected non-background sprite that is not a solid 2x2 square.
- There is one solid 2x2 square anchor in another non-background color.

Rule:
- Move the sprite straight toward the anchor along their unique cardinal separation until it touches the anchor's near edge.
- Reflect the moved sprite across that near edge and draw the reflected copy in the anchor color.
- Clear the original locations; the output keeps only the moved sprite and reflected anchor-color copy.

Key insight:
- The 2x2 square marks the mirror line: its side facing the sprite becomes the contact edge for a two-color reflected pair.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="eval" puzzle_id="97d7923e" lines="81-94">
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

<reference_puzzle index="8" source="train" puzzle_id="14b8e18c" lines="40-57">
<description>
Mark every complete square object with red pixels just outside its corners.

Input structure:
- A uniform background contains one or more monochrome 4-connected foreground objects.
- The relevant objects are square frames or filled squares: their bbox is square
  and all cells on the bbox border have the object's color.
- Non-square bars and incomplete fragments may also appear and are ignored.

Rule:
- Keep the input unchanged.
- For each complete square object, paint red on the two outside background
  pixels orthogonally adjacent to each of the square's four corners.

Key insight: The transformation is not based on size or color; it selects
objects whose bounding box is a complete square border and labels only their
exposed outer corner-adjacent cells.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="902510d5" lines="58-80">
<description>
Keep the largest 8-connected motif and replace the corner/control pixels with a solid corner triangle.

Input structure
- The background is the dominant color.
- Exactly one grid corner contains a foreground control pixel; it gives the
  corner from which the triangle is painted.
- There is one unique largest 8-connected monochrome foreground object, the
  motif that remains in the output; all other foreground pixels are controls.
- Every foreground component outside that motif is a singleton control pixel.
- The fill color is the unique control color that appears in at least two
  non-corner control pixels.

Rule
- Start from a blank background canvas and redraw only the largest motif.
- Count the non-corner control pixels; that count is the triangle side.
- Paint the fill color on the right-triangle region where those two distances
  from the marked corner's edges sum to less than the side length.

Key insight
- The scattered singleton controls are counted into a compact triangular
  stack at the marked corner; only the largest motif is not part of the count.
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

<reference_puzzle index="11" source="eval" puzzle_id="8f215267" lines="7-21">
<description>
Count same-color patterns per frame and represent the tally as dots inside.

Input structure:
- bg = most common color.
- Hollow rectangular frames in various colors.
- Small monochrome pattern sprites scattered outside frames.

Rule:
- Remove all small patterns from the grid.
- For each frame of color C, count how many patterns share color C.
- Draw that many dots along the frame's center row, spaced by 2, starting from the right.

Key insight: Frames act as tally displays — scattered same-color patterns are "absorbed" and their count is shown as interior dots.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="25e02866" lines="14-31">
<description>
Merge same-sized filled rectangular objects by overlaying their internal marker pixels.

Input structure:
- A dominant background surrounds two or more separated 4-connected foreground objects.
- Every object is a filled rectangle of the same size.
- Each rectangle has the same majority base color plus one or more non-base marker pixels.

Rule:
- Create one output rectangle with the shared object size, filled with the shared base color.
- For each input rectangle, copy every non-base marker pixel to the same relative location
  in the output.
- Marker overlays must either land on base color or agree in color with an existing marker.

Key insight:
The separated rectangles are layers of the same template; their colored marker pixels
combine by local coordinates while the repeated base fill is just the output canvas.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="afe3afe9" lines="106-119">
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

<reference_puzzle index="14" source="eval" puzzle_id="221dfab4" lines="6-19">
<description>
Propagate a repeating striped pattern from a yellow border segment toward the opposite edge, recoloring foreground objects on certain stripes.

Input structure:
- bg = most common color. Foreground blobs of another color scattered in the interior.
- A short yellow (4) segment sits on one border edge.

Rule:
- From the yellow segment, march toward the opposite edge in steps of 2 cells, cycling colors Y, Y, G (4, 4, 3). The 1-cell gap between stripes is forced to bg.
- Yellow stripes paint only within the segment's span.
- Green stripes additionally recolor all non-bg pixels across the entire row/column to green.

Key insight: Green stripes act as a "flood" that recolors existing objects on the full row/column, while yellow stripes are confined to the segment span.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="eval" puzzle_id="195c6913" lines="37-52">
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

<reference_puzzle index="16" source="train" puzzle_id="8e301a54" lines="8-23">
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

<reference_puzzle index="17" source="train" puzzle_id="37ce87bb" lines="33-47">
<description>
Compute a grey result bar by adding teal bar heights and subtracting red bar heights.

Input structure
- The grid has a dominant background and several bottom-aligned one-cell-wide vertical bars.
- Teal bars are positive operands; red bars are negative operands.
- The penultimate column is an empty result slot.

Rule
- Keep every input bar unchanged.
- Compute result height = total teal bar height - total red bar height.
- Paint that many bottom cells of the penultimate column grey.

Key insight
- The bars encode arithmetic by height, with color providing the sign.
</description>
</reference_puzzle>

<reference_puzzle index="18" source="eval" puzzle_id="c7f57c3e" lines="22-35">
<description>
Swap every sprite to the opposite prototype while preserving scale and position.

Input structure:
- bg = most common color. Two distinct small sprite patterns (prototypes) appear at various scales.
- Each sprite instance is a scaled copy of one of the two prototypes, placed on the grid.

Rule:
- Identify the two prototype patterns from the smallest-area sprites.
- For each sprite, determine which prototype it matches and its integer scale factor.
- Replace it with the other prototype at the same scale, aligned by an internal anchor region (color-1 pixels).

Key insight: Every sprite toggles to the alternate prototype; the anchor sub-region inside each sprite pins the alignment so swapped sprites land in the correct position regardless of scale.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="c4d1a9ae" lines="28-48">
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

<reference_puzzle index="20" source="eval" puzzle_id="247ef758" lines="35-46">
<description>
Move colored sprites from the left strip into a framed area at marker-defined targets.

Input structure:
- bg background; rectangular frame on right with colored markers on its edges; sprites in a narrow left strip separated by a constant-color divider column.

Rule:
- For each color, target rows/cols are where both opposite frame edges share that color. Targets are the Cartesian product.
- Erase each sprite from the left and stamp it centered at every target. Higher sprites cover lower ones.

Key insight: One sprite replicates to a grid of row × column marker intersections.
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
