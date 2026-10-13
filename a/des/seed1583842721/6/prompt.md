你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="train" puzzle_id="11dc524f" lines="25-40">
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

<reference_puzzle index="2" source="train" puzzle_id="412b6263" lines="39-54">
<description>
Infer the background, rotate the whole input 90 degrees counterclockwise, and stack two framed copies vertically.

Input structure:
- Blue is absent from the input.
- One rare marker color appears in exactly two pixels.
- The background is the unique non-marker color whose removal leaves exactly two 4-connected foreground objects, each with one marker pixel and one body color.
- The whole grid, including background and foreground objects, is the sprite to transform.

Rule:
- Rotate the entire input grid 90 degrees counterclockwise.
- Create a blue one-pixel frame around two vertical copies of the rotated grid.
- Use the input background color for the two endpoints of every horizontal frame/separator row.

Key insight: The output is not a crop of the foreground objects; it is two copies of the complete rotated input, separated and enclosed by a newly added blue border.
</description>
</reference_puzzle>

<reference_puzzle index="3" source="eval" puzzle_id="58490d8a" lines="9-21">
<description>
Count colored objects and record tallies inside a legend box.

Input structure:
- A black rectangular legend box with key-color pixels in its leftmost non-black column.
- Outside the box, distinct objects of each key color are scattered across the grid.

Rule:
- For each key color, count separate same-colored objects outside the legend box.
- Output the legend box repainted: each key row gets one dot per object, spaced every other cell rightward from the key pixel.

Key insight: The legend is both the color index and the output canvas — object counts become dot sequences on the matching row.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="9968a131" lines="10-27">
<description>
Shift alternating two-color row bars so each row aligns to the topmost bar's color columns.

Input structure:
- The dominant background surrounds horizontal two-pixel bars.
- Every foreground row has exactly one adjacent bar using the same two colors.
- The topmost bar gives the reference left/right column for each color.

Rule:
- Keep bars whose left color is already in its reference column.
- For a reversed bar, shift the whole two-pixel bar horizontally until its
  left color lands in that color's reference column.
- Background cells vacated by shifted bars become background.

Key insight:
- The rows are not recolored; each two-color bar is translated so equal
  colors line up with the topmost reference bar.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="494ef9d7" lines="16-30">
<description>
Slide each blue/teal or yellow/orange row pair together while preserving its left-to-right order.

Input structure
- The grid has a black background.
- Each nonempty row contains exactly two distinct foreground pixels.
- Blue/teal and yellow/orange row pairs are the moving pairs; other color pairs are distractors.

Rule
- For each moving pair, keep the left pixel fixed and move the right pixel left until it touches the left pixel.
- Clear the moved pixel's original location, and leave non-moving rows unchanged.

Key insight
- Only same-hue-family pixel pairs attract horizontally; separated unrelated colors do not compact.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="eval" puzzle_id="a32d8b75" lines="262-275">
<description>
Assemble and place enlarged sprites on a canvas using legend-panel instructions.

Input structure:
- Color-6 separators divide the grid into a left legend column and a main canvas.
- Each legend panel has (top to bottom): a unit sprite, a frame template, a 5x5 bordered square with a position-marker pixel, and a 3x3 rotation clock-hand.

Rule:
- Tile the unit sprite into each active cell of the frame to build an enlarged composite.
- Rotate the composite per the clock-hand direction (up=0, right=90, down=180, left=270 degrees).
- Place it on the canvas anchored at the position encoded by the marker pixel's row/col in the bordered square's 3x3 inner grid.

Key insight: The marker pixel's position in a 3x3 grid jointly encodes vertical and horizontal anchor alignment (start/middle/end).
</description>
</reference_puzzle>

<reference_puzzle index="7" source="eval" puzzle_id="db695cfb" lines="8-19">
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

<reference_puzzle index="8" source="train" puzzle_id="e84fef15" lines="39-59">
<description>
Reconstruct the repeated tile from its separated copies and mark the missing pixels in the damaged copy blue.

Input structure:
- The input is a lattice of equal-size blocks separated by one-cell solid rows
  and columns of the same separator color.
- All blocks are copies of one motif except exactly one block, where some
  motif pixels have been overwritten by the motif background color.
- Blue is absent from the input and is reserved as the output marker color.

Rule:
- Recover the motif by taking the unique majority color at each local block
  coordinate across all block copies.
- Find the single block that differs from this consensus motif.
- Return the consensus motif, recoloring the differing local coordinates blue.

Key insight:
- The repeated copies provide redundancy: the damaged positions are exactly
  the coordinates where the lone nonconforming copy has background instead of
  the consensus motif color.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="5792cb4d" lines="44-59">
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

<reference_puzzle index="10" source="train" puzzle_id="18286ef8" lines="17-31">
<description>
Use the pink control pixel to choose where the maroon pixel belongs inside the grey 3x3 region, and turn the control pixel maroon.

Input structure:
- The grid contains one 3x3 region made from eight grey pixels and one maroon pixel.
- There is exactly one pink control pixel outside that region.

Rule:
- Restore the old maroon cell in the 3x3 region to grey.
- Compare the pink pixel to the 3x3 region: above/same band/below selects top/middle/bottom, and left/same band/right selects left/middle/right.
- Paint the selected cell maroon and recolor the pink control pixel maroon.

Key insight:
- The pink pixel acts as a relative-position marker for a 3x3 target inside the grey region.
</description>
</reference_puzzle>

<reference_puzzle index="11" source="eval" puzzle_id="2d0172a1" lines="209-224">
<description>
Convert a rough nested line drawing into a clean canonical frame-and-dot diagram.

Input structure:
- The grid has one background color and one foreground color.
- Foreground 4-connected objects are rough closed frames or small filled blobs.
- Frames may contain other frames/blobs; blobs outside a frame remain outside it.

Rule:
- Detect the containment tree of foreground objects using holes of each frame object.
- Replace every blob with one foreground pixel.
- Replace every frame with a rectangular border padded two cells around its direct children.
- Place siblings by their original row/column overlap and order, with one empty cell between canonical objects.

Key insight: The exact wobbly geometry is irrelevant; only enclosure plus relative sibling positions define the output.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="train" puzzle_id="458e3a53" lines="19-32">
<description>
Extract the compact table of solid cells hidden inside a separator lattice.

Input structure
- The grid has full-width and full-height bars of one separator color, dividing it into rectangular cells.
- Most cells contain a repeating multicolor pattern, while a contiguous rectangle of cells is solid monochrome.

Rule
- Split the input into lattice cells using the full separator rows and columns.
- Keep the cells whose full interior is one color, crop to their occupied rectangle, and output those colors.

Key insight
- The periodic pattern is noise; only completely solid lattice cells encode the reduced output grid.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="37ce87bb" lines="33-47">
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

<reference_puzzle index="14" source="train" puzzle_id="b7955b3c" lines="386-403">
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

<reference_puzzle index="15" source="train" puzzle_id="17b866bd" lines="49-67">
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

<reference_puzzle index="16" source="eval" puzzle_id="332f06d7" lines="9-23">
<description>
Slide a solid square through a winding channel to reach a same-sized target square.

Input structure:
- bg (most common color). A river color forms a winding corridor.
- Two same-sized NxN solid squares on the river: mover and destination, each a distinct color.

Rule:
- The mover square travels along the river toward the destination.
- It can only pass through positions where the full NxN area is river.
- If reachable, the mover replaces the destination; otherwise it stops at the farthest reachable point.
- The mover's original position is filled with the river color.

Key insight: Chokepoints narrower than N pixels block the square, forcing it to stop short.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="1be83260" lines="41-58">
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

<reference_puzzle index="18" source="train" puzzle_id="50c07299" lines="8-21">
<description>
Move the diagonal foreground segment to the next longer run along the same up-right diagonal.

Input structure
The grid has one background color and one foreground color. The foreground pixels form a
contiguous one-pixel-wide segment on a single anti-diagonal.

Rule
Erase the current segment. If its columns are c0..c1 and its length is L, draw the same
color on columns c1+1..c1+L+1 along the same anti-diagonal.

Key insight
The segment advances past its current upper-right end by one whole segment position, and
the next segment is one pixel longer than the current one.
</description>
</reference_puzzle>

<reference_puzzle index="19" source="train" puzzle_id="da6e95e5" lines="70-81">
<description>
Extract the central object from the marker cluster whose markers have the same color.

Input structure
The grid has a dominant background and several clusters; each cluster has four
same-color corner marker objects around one central monochromatic object.
Rule
Find every central object enclosed by four aligned corner markers of one color,
then choose the unique one whose object color equals the marker color.
Key insight
The answer is not the largest or most frequent object; it is the center of the
self-colored cluster, cropped to that object's bounding box with background kept.
</description>
</reference_puzzle>

<reference_puzzle index="20" source="eval" puzzle_id="d35bdbdc" lines="120-134">
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
   - **允许使用**（较为简单的计算）：数像素 / 孔洞 / 对象 / 邻居（≤10 量级）、比大小、按某个可见属性排序、奇偶判断、取最大 / 最小、用一个计数值作为另一个等量几何量（如 N 个像素 → 长度 N 的射线、面积 N → 边长 N）、判断接触 / 相邻 / 包含、粗略中心位置（簇的视觉中点）、"距离"作为"几步"而不是公式。
   - **禁止使用**（这些是算法量而非视觉量，会迫使解题者依赖公式而非视觉直觉）：曼哈顿 / 欧氏 / Chebyshev 距离作为精确查表键、凸包面积、二进制位编码 / 位运算、模运算 / 余数、拓扑距离（BFS / 最短路 / 跳数）、精确几何质心坐标、凸角 / 凹角计数。
3. 问题必须能从少量示例中推断出来（通常为 3 对训练样例 + 1 个测试样例）。设计时从人类解题者的视觉角度思考：规则涉及的所有关键要素（对象、关系、属性）必须在网格上肉眼可见，不应依赖人眼无法直接观察到的隐含信息或需要推算的隐藏属性。规则背后的逻辑本身可以有一定挑战性，但解题所需的所有线索必须能从网格中直接获取。
4. 生成的问题描述旨在促进后续用于创建题目网格的代码生成。因此，对输入 / 输出网格的描述以及规则总结必须极其详细；语言风格应易于理解；对于特定对象（如箭头），必须在描述中明确其绘制方式。
5. 描述谜题时避免使用 0 或 1 这类颜色编号进行排序；应使用颜色名称，例如 "red"。可用颜色如下：Black、Blue、Red、Green、Yellow、Grey、Magenta、Orange、Azure、Maroon。
6. 排序或优先级规则必须基于网格中可见的属性（如位置、面积、数量），不得依赖颜色名称的字母顺序或颜色编号等先验知识。
7. 每个新题的核心推理机制应明显不同——即从观察样例到归纳规则的认知路径不同，而不仅仅是场景、对象或颜色不同。
8. 题目难度应来自机制的隐蔽性而非步骤数。用参考题作为难度标尺——如果你生成的题一个样例就能猜到规则，说明机制太透明，需要加深隐蔽性。
9. 优先从参考题的机制出发进行重组或推演，而非从物理隐喻出发。物理隐喻容易导致大量题目归结为一种模式，例如"沿方向移动到阻挡处停止"。
10. 题目的难度优先来自规则的**结构层数**，而不是单一步骤的复杂度。关于"结构层数"的判定标准：**只有解题者需要主动推理才能完成的步骤才算一层**。肉眼一扫即可获得的信息（如"找到缺口在哪一侧"、"数出像素个数"、"识别哪个对象最大"）不构成独立的推理层——它们只是变换的前置输入，不增加认知难度。真正的多层结构是指每一层都需要解题者发现一个非显然的规律。
   如果你发现自己写出的题只是"对每个对象，根据它的某个属性做某变换"的单步规则，请引入一个需要主动推理的前置步骤。**禁止用堆叠 tie-break 规则或复杂数学公式来"加难度"是劣解；用多一层需要归纳的结构嵌套来"加难度"才是好解**。
11. 层数可以是两层、三层或更多——不必拘泥于两层。若题目的机制天然适合更深的嵌套，鼓励向三层及以上尝试；但层数本身不是目的，每一层都必须有独立的归纳价值。
12. **严禁用复杂数学或复杂算法来制造层级深度或增加难度。** 这是最常见的失败模式：为了凑层数或提升难度，引入注意事项第 2 条禁止列表中的复杂计算（二进制编码、距离公式、模运算、BFS、全局最优匹配等）。简单计算（计数、比大小、奇偶判断等）作为层的一部分是允许的，但复杂计算本身不构成有价值的推理层，也不应被用来提升难度。例如：
   - 好的两层："颜色配对找到对应模板 → 碎片相对于模板的方位决定旋转角度"（两层都是视觉关系）
   - 坏的两层："数出内部标记个数 N → 将 N 转为二进制画出柱状图"（第二层是复杂数学运算，不是视觉发现）

生成后对生成的题执行以下六项审查（审查内容不需要保存到文件）：
1. 原创性
  新题不可与上述参考题相同或类似，不可只是替换颜色、方向、对象类型等表层要素。判断标准：新题的核心推理路径应与任何一道参考题不同。
2. 确定性
  规则完全确定，同一输入只能产生唯一输出，每一步操作都有明确定义。
3. 可实现性
  所有必要细节都已给出——对象如何生成、如何识别、如何变换、边界情况如何处理，读完描述后可以直接写代码。禁止使用上述禁用的数学知识和人类不可用的信息（颜色id排序等）。
4. ARC 合规性
  网格尺寸合理（≤30×30），使用颜色不超过 10 种，规则复杂度适中，人类能从 3-5 个样例中发现规律。
5. 数学/算法清洁性
  对每道题逐条检查：规则中是否隐含了注意事项第 2 条禁止列表中的复杂计算或复杂算法？简单计算（计数、比大小、奇偶等）不在此列，但需警惕以下常见的伪装形式：
  - 用"扩散""传播"包装的 BFS / 最短路
  - 用"编码""解码"包装的二进制转换
  - 用"优先级""竞争"包装的全局最优匹配
  - 用"模拟""时间步"包装的迭代仿真
  若发现任何一项，必须将该步骤替换为简单视觉机制，或删除该题重新生成。
6. 层级有效性与内部一致性
  逐题检查以下两点，发现问题需要重写该题：
  - **层级是否会坍缩**：每一层是否都对最终输出有独立贡献？是否存在某一层无论怎么变化都不影响下一层结果的情况？（例如规则号称"先用 A 选出 B，再用 B 控制 C"，但 A 的所有可能取值都导出同一个 B，第一层实际无效，整道题坍缩为单层。）
  - **层级是否"感知透明"**：某一层的结果是否无需从样例中归纳、仅靠直觉就能看出？（例如"数嵌套层数""找面积最大的色块""看缺口在哪一侧"等——这些是直接读取而非推理，不算独立的归纳层。真正的层应该是：如果只看单个样例，无法确定该层的规则，必须对比多个样例才能发现。）
  - **规则是否自相矛盾**：题目内部是否存在隐性冲突或未覆盖的退化情况？（例如两条优先级条件在某些输入下会冲突，或某个对象没有标记、某个区域无内部像素时规则失效。）

若任意一题未通过上述六项审查中的任何一项，丢弃该题并重新生成一道替代题，替代题同样需要通过全部六项审查后才可保留。
