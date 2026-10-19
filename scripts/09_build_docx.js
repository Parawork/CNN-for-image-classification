// Build the EN3150 A03 report as a Word document from the markdown source.
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, PageBreak,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle, ImageRun,
  LevelFormat, Footer, PageNumber, convertInchesToTwip,
} = require("docx");

const SRC = path.join(__dirname, "..", "report", "report_source.md");
const FIGROOT = "E:\\Pattern\\outputs\\figures";
const OUT = "E:\\Pattern\\submission\\PatternX_A03_EN3150.docx";

const CONTENT_DXA = 9026;        // A4 (11906) minus 1" margins both sides
const CONTENT_PX = 600;          // ~6.25in at 96dpi
const NAVY = "1F4E79";
const DARK = "15314B";

// ---------------------------------------------------------------- helpers
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

/** Markdown inline -> TextRun[] (**bold**, *italic*, `code`). */
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|(?<!\*)\*[^*]+\*(?!\*)|`[^`]+`)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) {
      out.push(new TextRun({ ...base, text: tok.slice(2, -2), bold: true }));
    } else if (tok.startsWith("`")) {
      out.push(new TextRun({ ...base, text: tok.slice(1, -1), font: "Consolas", size: 18 }));
    } else {
      out.push(new TextRun({ ...base, text: tok.slice(1, -1), italics: true }));
    }
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out.length ? out : [new TextRun({ text, ...base })];
}

const body = (text, opts = {}) => new Paragraph({
  children: runs(text, { font: "Times New Roman", size: 21 }),
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 140, line: 276 },
  ...opts,
});

const bullet = (text) => new Paragraph({
  children: runs(text, { font: "Times New Roman", size: 21 }),
  numbering: { reference: "report-bullets", level: 0 },
  alignment: AlignmentType.JUSTIFIED,
  spacing: { after: 90, line: 264 },
});

const heading = (text, level) => new Paragraph({
  children: runs(text, {
    font: "Calibri",
    size: level === 1 ? 30 : level === 2 ? 24 : 21,
    bold: true,
    italics: level === 3,
    color: level === 1 ? DARK : level === 2 ? NAVY : "333333",
  }),
  heading: level === 1 ? HeadingLevel.HEADING_1
    : level === 2 ? HeadingLevel.HEADING_2 : HeadingLevel.HEADING_3,
  spacing: { before: level === 1 ? 320 : level === 2 ? 240 : 180, after: level === 3 ? 90 : 140 },
});

function cellPara(text, { header = false, right = false } = {}) {
  return new Paragraph({
    children: runs(text, {
      font: header ? "Calibri" : "Times New Roman",
      size: 17,
      bold: header || undefined,
      color: header ? "FFFFFF" : undefined,
    }),
    alignment: right ? AlignmentType.RIGHT : AlignmentType.LEFT,
    spacing: { after: 0, line: 240 },
  });
}

/** rows: string[][] with the markdown header + separator already split out. */
function buildTable(header, aligns, bodyRows) {
  const n = header.length;
  const all = [header, ...bodyRows];

  // proportional widths, floored at the longest single word so nothing breaks mid-word
  const CHAR = 95;                       // ~DXA per character at 8.5pt
  const want = [], floor = [];
  for (let i = 0; i < n; i++) {
    const col = all.map((r) => r[i] ?? "");
    want.push(Math.max(4, ...col.map((c) => c.length)));
    const longestWord = Math.max(4, ...col.flatMap((c) => c.split(/\s+/).map((w) => w.length)));
    floor.push(Math.min(longestWord * CHAR + 160, (CONTENT_DXA / n) * 2.0));
  }
  const total = want.reduce((a, b) => a + b, 0);
  let widths = want.map((w) => (CONTENT_DXA * w) / total);
  for (let pass = 0; pass < n; pass++) {
    const deficit = widths.reduce((s, w, i) => s + Math.max(0, floor[i] - w), 0);
    if (deficit < 5) break;
    const slack = widths.map((w, i) => Math.max(0, w - floor[i]));
    const slackSum = slack.reduce((a, b) => a + b, 0);
    if (slackSum <= 0) break;
    const k = Math.min(1, deficit / slackSum);
    widths = widths.map((w, i) => Math.max(w, floor[i]) - slack[i] * k);
  }
  const sum = widths.reduce((a, b) => a + b, 0);
  widths = widths.map((w) => Math.round((CONTENT_DXA * w) / sum));
  widths[n - 1] += CONTENT_DXA - widths.reduce((a, b) => a + b, 0);   // exact sum

  const mkRow = (cells, isHeader, idx) => new TableRow({
    tableHeader: isHeader,
    children: cells.map((c, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      shading: {
        type: ShadingType.CLEAR,
        fill: isHeader ? NAVY : (idx % 2 ? "F2F6FA" : "FFFFFF"),
        color: "auto",
      },
      margins: { top: 50, bottom: 50, left: 70, right: 70 },
      children: [cellPara(c, { header: isHeader, right: !isHeader && aligns[i] === "right" })],
    })),
  });

  return new Table({
    columnWidths: widths,
    width: { size: CONTENT_DXA, type: WidthType.DXA },
    rows: [mkRow(header, true, 0), ...bodyRows.map((r, i) => mkRow(r, false, i + 1))],
  });
}

function figure(caption, file) {
  const full = path.join(FIGROOT, path.basename(file));
  const { w, h } = pngSize(full);
  const scale = Math.min(CONTENT_PX / w, 430 / h);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 60 },
      children: [new ImageRun({
        type: "png",
        data: fs.readFileSync(full),
        transformation: { width: Math.round(w * scale), height: Math.round(h * scale) },
      })],
    }),
    new Paragraph({
      children: runs(caption, { font: "Times New Roman", size: 17, color: "444444" }),
      alignment: AlignmentType.CENTER,
      spacing: { after: 220 },
    }),
  ];
}

// ---------------------------------------------------------------- parse
const md = fs.readFileSync(SRC, "utf8");
const [coverSrc, ...rest] = md.split(/^---$/m);
const bodySrc = rest.join("---");

function parse(src) {
  const lines = src.split("\n");
  const flow = [];
  let para = [];
  const flush = () => { if (para.length) { flow.push(body(para.join(" "))); para = []; } };

  for (let i = 0; i < lines.length; i++) {
    const s = lines[i].trim();
    if (!s) { flush(); continue; }
    if (s === "---") {
      flush();
      flow.push(new Paragraph({ children: [new PageBreak()] }));
      continue;
    }
    if (s.startsWith("### ")) { flush(); flow.push(heading(s.slice(4), 3)); continue; }
    if (s.startsWith("## ")) { flush(); flow.push(heading(s.slice(3), 2)); continue; }
    if (s.startsWith("# ")) { flush(); flow.push(heading(s.slice(2), 1)); continue; }
    const img = s.match(/^!\[(.*)\]\((.*)\)$/);
    if (img) { flush(); flow.push(...figure(img[1], img[2])); continue; }
    if (s.startsWith("- ")) { flush(); flow.push(bullet(s.slice(2))); continue; }
    if (s.startsWith("|")) {
      flush();
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) {
        rows.push(lines[i].trim().replace(/^\||\|$/g, "").split("|").map((c) => c.trim()));
        i++;
      }
      i--;
      const header = rows[0];
      let aligns = header.map(() => "left");
      let bodyRows = rows.slice(1);
      if (bodyRows.length && bodyRows[0].every((c) => /^:?-+:?$/.test(c))) {
        aligns = bodyRows[0].map((c) => (c.endsWith(":") ? "right" : "left"));
        bodyRows = bodyRows.slice(1);
      }
      flow.push(buildTable(header, aligns, bodyRows));
      flow.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
      continue;
    }
    para.push(s);
  }
  flush();
  return flow;
}

// ---------------------------------------------------------------- cover
const centered = (text, opts = {}) => new Paragraph({
  children: runs(text, { font: "Calibri", size: 22, ...(opts.run || {}) }),
  alignment: AlignmentType.CENTER,
  spacing: { after: opts.after ?? 120 },
});

const coverFlow = [
  new Paragraph({ spacing: { after: 1800 }, children: [] }),
  centered("EN3150 \u2013 Pattern Recognition", { run: { size: 26, color: NAVY }, after: 220 }),
  centered("Assignment 03", { run: { size: 40, bold: true, color: DARK }, after: 60 }),
  centered("Resource-Constrained CNN for Edge Image Classification",
    { run: { size: 30, bold: true, color: DARK }, after: 560 }),
  centered("Group PatternX", { run: { size: 24, bold: true }, after: 240 }),
];

const members = [
  ["Index number", "Name"],
  ["220303E", "Kariyawasam U.G.R.T."],
  ["220370E", "Maddumage M.M.T.D."],
  ["220522A", "Rathnayaka R.M.P.B."],
  ["220525K", "Rathnayake R.M.D.D."],
];
const memberWidths = [2600, 4400];
coverFlow.push(new Table({
  columnWidths: memberWidths,
  width: { size: 7000, type: WidthType.DXA },
  alignment: AlignmentType.CENTER,
  rows: members.map((r, idx) => new TableRow({
    tableHeader: idx === 0,
    children: r.map((c, i) => new TableCell({
      width: { size: memberWidths[i], type: WidthType.DXA },
      shading: {
        type: ShadingType.CLEAR,
        fill: idx === 0 ? NAVY : (idx % 2 ? "FFFFFF" : "F2F6FA"),
        color: "auto",
      },
      margins: { top: 70, bottom: 70, left: 110, right: 110 },
      children: [new Paragraph({
        children: runs(c, {
          font: idx === 0 ? "Calibri" : "Times New Roman",
          size: 21,
          bold: idx === 0 || undefined,
          color: idx === 0 ? "FFFFFF" : undefined,
        }),
        alignment: AlignmentType.CENTER,
        spacing: { after: 0 },
      })],
    })),
  })),
}));

coverFlow.push(
  new Paragraph({ spacing: { after: 420 }, children: [] }),
  centered("Repository: https://github.com/Parawork/Pattern", { run: { size: 21 }, after: 80 }),
  centered("Dataset: EuroSAT (RGB), 27,000 images, 10 classes, 64\u00d764",
    { run: { size: 21 }, after: 480 }),
  centered("Results at a glance", { run: { size: 22, bold: true }, after: 160 }),
);

const glance = [
  ["", "Params", "Size", "MACs", "Test accuracy"],
  ["Model A (standard CNN)", "405,994", "1.56 MB", "60.18 M", "96.94%"],
  ["Model B (lightweight CNN)", "63,659", "265 KB", "7.33 M", "94.96%"],
  ["ShuffleNetV2 x1.0 (fine-tuned)", "1,263,854", "4.99 MB", "11.76 M", "97.93%"],
  ["MobileNetV2 (fine-tuned)", "2,236,682", "8.77 MB", "24.46 M", "98.54%"],
];
coverFlow.push(buildTable(glance[0], ["left", "right", "right", "right", "right"], glance.slice(1)));
coverFlow.push(new Paragraph({ children: [new PageBreak()] }));

// ---------------------------------------------------------------- document
const doc = new Document({
  creator: "Group PatternX",
  title: "EN3150 Assignment 03 - Resource-Constrained CNN for Edge Image Classification",
  description: "EN3150 Assignment 03 report",
  numbering: {
    config: [{
      reference: "report-bullets",
      levels: [{
        level: 0,
        format: LevelFormat.BULLET,
        text: "\u2022",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 420, hanging: 220 } } },
      }],
    }],
  },
  sections: [{
    properties: {
      page: { margin: {
        top: convertInchesToTwip(1), bottom: convertInchesToTwip(1),
        left: convertInchesToTwip(1), right: convertInchesToTwip(1),
      } },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({
            children: ["EN3150 Assignment 03 \u2013 Group PatternX  |  Page ", PageNumber.CURRENT],
            font: "Calibri", size: 16, color: "777777",
          })],
          border: { top: { style: BorderStyle.SINGLE, size: 4, color: "DDDDDD", space: 6 } },
        })],
      }),
    },
    children: [...coverFlow, ...parse(bodySrc)],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT, buf);
  console.log("wrote", OUT, (buf.length / 1024).toFixed(0) + " KB");
});
