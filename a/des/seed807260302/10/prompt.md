你是一个 ARC-AGI 谜题设计专家。下面的 20 道参考题是难度和机制质量的标杆。仔细阅读每道题的**解题逻辑**（解题者从样例归纳规则的认知过程），而非表面场景。你的新题应在机制深度和隐蔽性上与参考题持平或者实现超越。

每个参考题描述包含：
- `description`：一句话概述题目的变换目标。
- `Input structure`：输入网格中有哪些对象、背景、标记、边界或组件。
- `Rule`：从输入到输出的确定性变换规则。
- `Key insight`：解题所需的核心洞见。

参考题目如下：
<reference_puzzle index="1" source="eval" puzzle_id="3dc255db" lines="37-53">
<description>
Eject content pixels from cup-shaped frames outward along each cup's tail direction.

Input structure:
- bg = background. Hollow cup/Y/funnel frames, each a unique border color.
- Each cup has a convergent tail spike on one side and an open mouth on the opposite side.
- Content pixels (a single non-border color per cup) sit inside the cup's mouth interior.

Rule:
- For each cup, identify its tail spike and tail direction.
- Erase all content pixels from the mouth interior.
- Draw a straight line of content-colored pixels from the spike outward in the tail direction,
  one pixel per erased content pixel; cells beyond the grid edge are dropped.

Key insight: The tail spike is the unique convergent point whose outward direction is opposite
the mouth opening where the content lives.
</description>
</reference_puzzle>

<reference_puzzle index="2" source="train" puzzle_id="11dc524f" lines="25-40">
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

<reference_puzzle index="3" source="train" puzzle_id="22208ba4" lines="12-27">
<description>
Move the repeated boundary color's rectangles one block inward while leaving singleton boundary rectangles in place.

Input structure:
- The background is the most common color.
- Foreground objects are solid monochrome rectangles touching a grid corner.
- Exactly one foreground color appears as multiple separate 4-connected objects; other colors appear once.

Rule:
- Clear the multiple objects of the repeated color from their original positions.
- For each cleared rectangle, translate it away from the row and column border it touches by exactly its height and width.
- Keep singleton-color objects unchanged; overlapping translated rectangles of the same color merge.

Key insight:
- Multiplicity selects the moving color, and each boundary rectangle's touched edges determine its inward diagonal translation.
</description>
</reference_puzzle>

<reference_puzzle index="4" source="eval" puzzle_id="71e489b6" lines="7-19">
<description>
Highlight black defects in blue foreground by surrounding them with orange.

Input structure:
- Blue (1) foreground on black (0) bg; small black holes or protrusions within blue regions; occasional thin blue spurs into black areas.

Rule:
- Thin blue pixels mostly surrounded by black are absorbed into the background.
- Small black components (≤3 cells) and leaf pixels of larger black components are "defects."
- Each defect gets an orange (7) 8-neighbor halo; the defect pixels themselves stay black.

Key insight: Only structural anomalies—tiny isolated patches and dangling endpoints—are marked, not bulk black regions.
</description>
</reference_puzzle>

<reference_puzzle index="5" source="train" puzzle_id="8fff9e47" lines="8-25">
<description>
Turn a 4-cell-wide strip of 2x2 blocks into four nested corner-fill patterns.

Input structure
The input is an even-sized rectangle whose shorter dimension is four pixels.
It can be partitioned exactly into 2x2 blocks, giving one block for each layer
of the output quadrants.

Rule
Read the 2x2 blocks in reverse row-major order: bottom-right block first,
moving left, then moving upward. For each block, send its top-left,
top-right, bottom-left, and bottom-right pixels to the layer sequences of the
upper-left, upper-right, lower-left, and lower-right output quadrants.

Key insight
Each input 2x2 block supplies the four colors for one concentric layer, and
the layer order is the block order from bottom-right back to top-left.
</description>
</reference_puzzle>

<reference_puzzle index="6" source="train" puzzle_id="b74ca5d1" lines="157-172">
<description>
Swap each 5x5 sprite with its corner marker color and stamp same-marker composite masks into the matching corner.

Input structure:
- A 30x30 grid has one dominant background color.
- Non-background corner pixels are marker labels.
- Every other foreground color is the body of one 5x5 sprite containing exactly one pixel of a corner marker color.

Rule:
- For each sprite, recolor body pixels to the sprite marker color and recolor the marker pixel back to its body color.
- If a marker lies on the side facing its matching corner and another same-marker sprite has a body pixel at that same local cell, use that visible body color for the marker pixel.
- For each corner marker, overlay the masks of all sprites with that marker color into the corner 5x5 block and paint the visible composite with the marker color.

Key insight:
The corner pixels are labels that group 5x5 sprites by their embedded marker color; each labeled corner shows the grouped sprite silhouette.
</description>
</reference_puzzle>

<reference_puzzle index="7" source="train" puzzle_id="8886d717" lines="43-63">
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

<reference_puzzle index="8" source="eval" puzzle_id="d59b0160" lines="39-54">
<description>
Delete connected components whose colors match a legend swatch.

Input structure:
- Background (bg) fills most of the grid.
- Top-left corner: a small square legend region bordered on its bottom and right edges by a single color; the interior holds a set of "target" colors.
- Elsewhere: multicolored objects (4-connected components) on the bg.

Rule:
- Read the target color set from inside the legend border.
- For each connected component, check whether it contains **all** target colors.
- Erase (fill with bg) every component that does; leave all others unchanged.
- Output grid is the same size.

Key insight: The legend acts as a filter specification — only components carrying the full color palette shown in the legend are removed.
</description>
</reference_puzzle>

<reference_puzzle index="9" source="train" puzzle_id="e45ef808" lines="31-46">
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

<reference_puzzle index="10" source="train" puzzle_id="689c358e" lines="72-88">
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

<reference_puzzle index="11" source="train" puzzle_id="83b6b474" lines="95-111">
<description>
Translate the colored objects into the most balanced complete rectangular frame.

Input structure
A grid with one dominant background color and several monochromatic
4-connected foreground objects that are straight bars or corner-like line
sprites.

Rule
Preserve every foreground object exactly, allowing translation only, and
place the objects so they cover every pixel of a one-cell-thick rectangular
frame exactly once; the frame interior is the background color.

Key insight
The output shape is not copied from the input: the foreground pixel count
determines possible frame sizes, and the intended output is the valid frame
assembly with the least difference between height and width.
</description>
</reference_puzzle>

<reference_puzzle index="12" source="eval" puzzle_id="e8686506" lines="172-185">
<description>
Assemble scattered colored fragments into a left-right symmetric rectangle, jigsaw-style.

Input structure:
- bg = most common color; remaining colors form small sprite fragments.
- Optional roles: 4 singleton pixels of one color mark corners; horizontal bars mark top/bottom edges; paired vertical lines mark left/right edges.

Rule:
- Collect all non-bg connected components as jigsaw pieces.
- Determine output rectangle dimensions from total piece area.
- Pack pieces into the rectangle so it is exactly mirror-symmetric (left-right) and same-color pieces never 4-touch.

Key insight: Left-right symmetry plus the no-same-color-touching constraint make the assembly unique; corner/edge markers anchor it when present.
</description>
</reference_puzzle>

<reference_puzzle index="13" source="train" puzzle_id="f18ec8cc" lines="71-86">
<description>
Reorder the vertical color-major blocks while preserving every block's internal pixels.

Input structure
The grid is divided into vertical blocks, each with a unique dominant color.
Small marker pixels from other block colors may appear inside the blocks.

Rule
If adjacent shared-color marker rows read exactly as the current block order
from right to left, reverse the blocks. Otherwise move the leftmost block to
the far right and shift the remaining blocks left.

Key insight
The transformation acts on whole vertical blocks, so marker pixels move with
their source block instead of being recolored or independently rearranged.
</description>
</reference_puzzle>

<reference_puzzle index="14" source="eval" puzzle_id="135a2760" lines="6-19">
<description>
Repair noisy periodic patterns inside frame-delimited grid regions.

Input structure:
- Frame color (most common) forms full rows/cols, dividing the grid into rectangular regions.
- bg (second most common) fills empty space; each region contains a small colored tile repeated periodically with a few corrupted pixels.

Rule:
- For each region, find the bounding box of non-bg content.
- Infer the smallest repeating tile that explains the pattern (majority vote per residue position), requiring it repeat at least 2x per axis.
- Replace the region with the clean tiled version.

Key insight: Tile dimensions are unknown and must be inferred; prefer the smallest tile with genuine repetition over one that memorizes the entire region.
</description>
</reference_puzzle>

<reference_puzzle index="15" source="train" puzzle_id="df978a02" lines="57-72">
<description>
Let the largest center-pointing object grow outward while every other object loses its inward tip.

Input structure
- The grid has a single dominant background color and four 8-connected monochromatic foreground objects.
- The objects surround a central background gap and point toward it from cardinal directions.
- Exactly one object has the largest area.

Rule
- Find the central background gap by minimizing aggregate distance to the four objects.
- Recolor the nearest-to-center pixel of each non-largest object to background.
- Add a three-pixel cap in the largest object's color on the side away from the center, centered on the shared row or column.

Key insight
- The edit is a transfer of prominence around a common center: losers retract at their tips, and the unique largest object extends outward from the center.
</description>
</reference_puzzle>

<reference_puzzle index="16" source="eval" puzzle_id="221dfab4" lines="6-19">
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

<reference_puzzle index="17" source="train" puzzle_id="3cd86f4f" lines="6-18">
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

<reference_puzzle index="18" source="train" puzzle_id="34cfa167" lines="94-117">
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

<reference_puzzle index="19" source="train" puzzle_id="2f767503" lines="66-81">
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

<reference_puzzle index="20" source="train" puzzle_id="e3fe1151" lines="36-55">
<description>
Fill the one blank in each corner region so all four regions have the same color multiset.

Input structure:
- The dominant background color forms one solid separator row and one solid
  separator column.
- The separator cross splits the grid into four equal rectangular corner regions.
- Each corner region contains exactly one background-colored blank pixel.

Rule:
- Count the non-background colors in each corner region.
- The completed region multiset is defined by the maximum observed count of
  each color across the four regions.
- Replace each region's blank pixel with the single color needed to match that
  completed multiset.

Key insight:
- The separator color inside a corner region is a missing value marker; comparing
  the four regions reveals the shared multiset without using numeric color order.
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

