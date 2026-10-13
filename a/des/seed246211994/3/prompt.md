你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="88e364bc" lines="102-115">
<description>
Slide yellow pixels inside walled regions in the direction dictated by border-legend arrows.

Input structure:
- bg = most common color; yellow pixels sit inside enclosed regions bounded by colored walls.
- Border legends: small rectangles on the grid edge whose fill color C matches a region's border color. Each legend contains a center dot and an indicator dot encoding a direction.

Rule:
- For each enclosed region, identify its border color C and look up the legend with fill color C.
- The direction from the legend's center dot to its indicator dot gives the slide direction.
- All yellow pixels in that region slide in that direction, stacking against the far wall.

Key insight: Legend fill color links to region border color; the center→indicator offset encodes the movement vector.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="60d73be6" lines="22-39">
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

<reference_puzzle index="3" source="eval" puzzle_id="67e490f4" lines="8-22">
<description>
Fill holes inside a large template frame using shape-matched outside sprites.

Input structure:
- bg = most common color. One large monochrome frame object (template).
- Inside the frame: holes of various shapes (non-frame-color regions).
- Outside the frame: small monochrome sprites scattered on bg.

Rule:
- Crop to the frame's bounding box; fill entirely with frame color.
- For each hole, find an outside sprite whose shape matches (rotation/flip allowed).
- Fill the hole with the matched sprite's color. If multiple sprites match, use the most frequent color.

Key insight: Sprites act as shape-coded color keys — their shape identifies which hole they fill, their color determines the fill color.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="train" puzzle_id="2faf500b" lines="30-44">
<description>
Remove each fracture marker line and push the surrounding object pixels one cell away from it.

Input structure
- A background grid with 8-connected foreground objects made from a body color and a rarer marker color.
- Each object has a non-square bbox; the marker pixels form one complete fracture line across the shorter bbox axis.

Rule
- Erase all marker pixels.
- For a wide object, each row has one marker; body pixels left of it shift one cell left and body pixels right of it shift one cell right.
- For a tall object, each column has one marker; body pixels above it shift one cell up and body pixels below it shift one cell down.

Key insight
- The marker line is a crack through the object, so the two sides separate by one pixel perpendicular to the crack.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="5751f35e" lines="29-43">
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

<reference_puzzle index="6" source="train" puzzle_id="5ad8a7c0" lines="10-24">
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

<reference_puzzle index="7" source="train" puzzle_id="c9680e90" lines="52-67">
<description>
Move each mirrored bottom source pixel to the far end of its trail and redraw the top half as its reflection.

Input structure
- A solid non-background row separates equal-height top and bottom halves.
- The top half has one marker color on the background.
- In the bottom half, exactly one color has positions that mirror the top markers across the separator, and the other color forms trails touching those source pixels.

Rule
- Identify the bottom source color by reflecting each bottom color into the top half and matching the top marker set.
- For every bottom source pixel, follow its unique 4-connected trail of the other bottom color to the trail endpoint.
- Clear the old top markers, bottom sources, and trails; place the source color at each endpoint and place the top marker color at the reflected endpoint.

Key insight
- The top half is only a mirror display of the bottom moving sources; the actual motion is encoded by the trail attached to each bottom source pixel.
</description>
</reference_puzzle>

<reference_puzzle index="8" source="train" puzzle_id="b71a7747" lines="10-23">
<description>
Extract the hidden two-color image by compacting the rows and columns of its complete mask.

Input structure
The grid contains one pair of colors whose pixels occupy every cell in the
cross product of their active rows and active columns; all other colors form
the surrounding filler or separators.
Rule
Find that unique color pair, keep exactly the rows and columns touched by
the pair, and return the induced subgrid.
Key insight
The target image is not chosen by color id; it is the only two-color mask
whose occupied pixels form a full rectangular row/column product after
deleting unused rows and columns.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="5034a0b5" lines="16-32">
<description>
Shift each interior pixel whose color matches a frame side one cell toward that matching side.

Input structure
The grid has a one-cell-thick rectangular frame: the four corners are the
background color, and each side between corners is a solid, distinct side
color. The interior is mostly background with sparse foreground pixels.

Rule
Interior pixels with the top-side color move one row up, bottom-side color
move one row down, left-side color move one column left, and right-side color
move one column right. A pixel already adjacent to its matching side stays in
place; colors that do not match a frame side and the frame itself are unchanged.

Key insight
The colored frame sides act as direction labels for matching interior pixels.
</description>
</reference_puzzle>

<reference_puzzle index="10" source="train" puzzle_id="42f83767" lines="23-39">
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

<reference_puzzle index="12" source="eval" puzzle_id="1818057f" lines="6-19">
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

<reference_puzzle index="13" source="train" puzzle_id="9841fdad" lines="15-39">
<description>
Expand the colored runs from the left reference panel into the blank right panel.

Input structure:
- A uniform rectangular frame surrounds the grid.
- One full-height divider of the frame color splits a left reference panel
  from a blank right panel.
- Both panels have one background margin column on each side of their active
  columns; the left active columns contain two-column endpoint slots, possibly
  connected by same-color horizontal runs.

Rule:
- Keep the original grid and paint only in the right panel's active columns.
- A source run exactly matching the left endpoint slot is copied to the left
  two columns of the right active span.
- A source run exactly matching the right endpoint slot is copied to the right
  two columns of the right active span.
- A source run connecting the endpoint slots is stretched across the entire
  right active span.

Key insight:
The left panel is a compressed legend for the right panel: endpoint markers
stay two pixels wide, while connectors between endpoints expand to fill the
distance between them.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="2d0172a1" lines="209-224">
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

<reference_puzzle index="15" source="eval" puzzle_id="58490d8a" lines="9-21">
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

<reference_puzzle index="16" source="train" puzzle_id="5a719d11" lines="44-59">
<description>
Within each horizontal pair of two-color panels, swap the motif masks and recolor each pasted mask with the source panel's base color.

Input structure:
- A separator color forms full blank rows and at least one full blank column.
- The non-separator area is split into row bands, each containing exactly two same-shaped rectangular panels.
- Each panel has a dominant base color and one minority motif color.

Rule:
- For every row band, clear each panel to its own base color.
- Paste the right panel's motif mask into the left panel using the right base color.
- Paste the left panel's motif mask into the right panel using the left base color.

Key insight:
- The motif colors themselves are distractors; only each motif's shape mask and the two panel base colors are exchanged.
</description>
</reference_puzzle>

<reference_puzzle index="17" source="train" puzzle_id="11dc524f" lines="25-40">
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

<reference_puzzle index="18" source="train" puzzle_id="6d1d5c90" lines="8-23">
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

<reference_puzzle index="19" source="eval" puzzle_id="58f5dbd5" lines="20-34">
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

<reference_puzzle index="20" source="train" puzzle_id="d968ffd4" lines="15-28">
<description>
Extend each of two aligned rectangular marker objects toward the other, filling half of the gap across the full perpendicular span.

Input structure
- A dominant background contains exactly two solid, same-sized monochromatic rectangles.
- The rectangles share either the same row span or the same column span, with a background gap between their facing edges.

Rule
- If the rectangles are side-by-side, color the left half of the inter-object gap across every row with the left object's color, and the right half with the right object's color.
- If one rectangle is above the other, color the upper half of the gap across every column with the upper object's color, and the lower half with the lower object's color.
- When the gap length is odd, leave the center row/column as background.

Key insight
- The marker objects are endpoints of an axial corridor; their own size identifies the facing edge, while the filled corridor spans the whole grid perpendicular to that axis.
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
