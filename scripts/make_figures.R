#!/usr/bin/env Rscript
# scripts/make_figures.R
#
# Main-text figures, drawn only from the tidy CSVs written by export_figure_data.py
# (no number is typed in this file).
#
#   Fig 1  pipeline and the three points at which patient information can leak
#   Fig 2  flow of events, recordings and participants
#   Fig 3  discrimination and calibration
#   Fig 4  comparator ladder
#   Fig 5  leakage experiment
#   Fig 6  model-derived attribution
#
# Output: PDF (vector) + 300 dpi LZW TIFF + PNG preview, 170 mm wide.
# Colour encodes the OUTCOME (Okabe-Ito); arms and classes use shape and line type.

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr); library(patchwork)
  library(scales); library(grid)
})

args <- commandArgs(trailingOnly = TRUE)
DATA <- if (length(args) > 0) args[1] else "results_clean/figure_data"
OUT  <- if (length(args) > 1) args[2] else "figures"
ONLY <- if (length(args) > 2) strsplit(args[3], ",")[[1]] else as.character(1:6)
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

MM <- 1 / 25.4
W  <- 170 * MM
TASKS <- c("Screening", "Sound pattern", "Disease group")
COL <- c("Screening" = "#0072B2", "Sound pattern" = "#D55E00", "Disease group" = "#009E73")
INK <- "#1A1A1A"; MUTED <- "#6B6B6B"; GRIDC <- "#E6E6E6"
PANEL <- "#F5F5F3"; RED <- "#B3261E"

rd <- function(f) read.csv(file.path(DATA, f), check.names = FALSE)
fct <- function(x) factor(x, levels = TASKS)

theme_pub <- function(base = 7) {
  theme_minimal(base_size = base, base_family = "sans") +
    theme(
      text             = element_text(colour = INK),
      plot.title       = element_text(size = base + 0.5, hjust = 0, face = "plain",
                                      margin = margin(b = 3)),
      plot.title.position = "plot",
      axis.title       = element_text(size = base),
      axis.text        = element_text(size = base - 0.5, colour = MUTED),
      axis.line        = element_line(colour = MUTED, size = 0.25),
      axis.ticks       = element_line(colour = MUTED, size = 0.25),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = GRIDC, size = 0.25),
      strip.text       = element_text(size = base, hjust = 0,
                                      margin = margin(b = 2)),
      legend.key.size  = unit(3.2, "mm"),
      legend.text      = element_text(size = base - 0.5),
      legend.title     = element_blank(),
      legend.margin    = margin(0, 0, 0, 0),
      plot.margin      = margin(2, 2, 2, 2),
      panel.spacing    = unit(3.2, "mm")
    )
}
theme_blank <- function() {
  theme_void(base_size = 7, base_family = "sans") +
    theme(plot.margin = margin(2, 2, 2, 2), text = element_text(colour = INK))
}
tag_theme <- theme(plot.tag = element_text(size = 9, face = "bold", hjust = 0),
                   plot.tag.position = c(0, 1))

save_fig <- function(p, name, h_mm) {
  h <- h_mm * MM
  ggsave(file.path(OUT, paste0(name, ".pdf")), p, width = W, height = h, units = "in",
         device = cairo_pdf)
  ggsave(file.path(OUT, paste0(name, ".png")), p, width = W, height = h, units = "in",
         dpi = 200, bg = "white")
  tiff(file.path(OUT, paste0(name, ".tiff")), width = W, height = h, units = "in",
       res = 300, compression = "lzw", type = "cairo", bg = "white")
  print(p); invisible(dev.off())
  cat(sprintf("  %s: %.1f MB tiff\n", name,
              file.size(file.path(OUT, paste0(name, ".tiff"))) / 1e6))
}

# helper: rounded box with wrapped text
boxes <- function(d, fill = PANEL, colour = MUTED, size = 2.1, lineheight = 1.05) {
  list(
    geom_tile(data = d, aes(x = x, y = y, width = w, height = h),
              fill = fill, colour = colour, size = 0.25),
    geom_text(data = d, aes(x = x, y = y, label = label), size = size,
              lineheight = lineheight, colour = INK)
  )
}
arrows <- function(d, colour = MUTED) {
  geom_segment(data = d, aes(x = x, y = y, xend = xend, yend = yend),
               colour = colour, size = 0.3,
               arrow = arrow(length = unit(1.1, "mm"), type = "closed"))
}

# ---------------------------------------------------------------- Fig 1: leak points
fig1 <- function() {
  step <- data.frame(
    x = c(0.10, 0.30, 0.50, 0.70, 0.90), y = 0.62, w = 0.185, h = 0.34,
    label = c("Annotated\nevent",
              "2 s clip\ncentred,\nband-passed",
              "Frozen HeAR\n+ LoRA\nadapters",
              "Three task\nheads",
              "LightGBM\n+ age, sex,\nsite"))
  seg <- data.frame(x = c(0.1925, 0.3925, 0.5925, 0.7925), y = 0.62,
                    xend = c(0.2075, 0.4075, 0.6075, 0.8075), yend = 0.62)
  leak <- data.frame(x = c(0.20, 0.50, 0.70), lab = c("a", "b", "c"))

  ggplot() +
    boxes(step, size = 2.3) + arrows(seg) +
    geom_point(data = leak, aes(x = x, y = 0.40), shape = 25, size = 2.4,
               fill = RED, colour = RED) +
    geom_text(data = leak, aes(x = x, y = 0.24, label = lab), size = 3.2,
              fontface = "bold", colour = RED) +
    coord_cartesian(xlim = c(0.0, 1.0), ylim = c(0.12, 0.85), expand = FALSE) +
    theme_blank()
}

# ---------------------------------------------------------------- Fig 2: flow + why it leaks
fig2 <- function() {
  m <- rd("flow_main.csv"); ex <- rd("flow_excluded.csv")
  ho <- rd("flow_holdout.csv"); fo <- rd("flow_folds.csv"); per <- rd("events_per_patient.csv")
  lab <- function(i, title) sprintf("%s\n%s events  |  %s recordings  |  %s participants",
                                    title, comma(m$events[i]), comma(m$recordings[i]),
                                    comma(m$patients_with_id[i]))
  main <- data.frame(x = 0.27, y = c(0.90, 0.66, 0.42), w = 0.5, h = 0.13,
                     label = c(lab(1, "Annotated events with audio"),
                               lab(2, "Events with a task label"),
                               lab(3, "Analysis cohort")))
  seg <- data.frame(x = 0.27, y = c(0.828, 0.588), xend = 0.27, yend = c(0.732, 0.492))
  exy <- c(0.785, 0.565, 0.495)
  extxt <- sprintf("%s\n%s events%s", ex$reason, comma(ex$events),
                   ifelse(nzchar(ex$detail), paste0("; ", ex$detail), ""))
  hb <- data.frame(x = 0.075 + (seq_len(nrow(ho)) - 1) * 0.145, y = 0.145, w = 0.135, h = 0.13,
                   label = sprintf("%s\n%s participants\n%s events", ho$partition,
                                   comma(ho$patients), comma(ho$events)))
  fb <- data.frame(x = 0.585 + (seq_len(nrow(fo)) - 1) * 0.088, y = 0.145, w = 0.082, h = 0.13,
                   label = sprintf("Fold %d\n%d", fo$fold, fo$patients))

  pa <- ggplot() +
    boxes(main[1:2, ]) + boxes(main[3, , drop = FALSE], fill = "white") +
    boxes(hb, fill = "white") + boxes(fb, fill = "white", size = 1.9) +
    arrows(seg) + arrows(data.frame(x = 0.27, y = exy, xend = 0.53, yend = exy)) +
    arrows(data.frame(x = 0.15, y = 0.355, xend = 0.15, yend = 0.215)) +
    geom_segment(data = data.frame(x = 0.52, y = 0.42, xend = 0.975, yend = 0.42),
                 aes(x = x, y = y, xend = xend, yend = yend), colour = MUTED, size = 0.3) +
    arrows(data.frame(x = 0.975, y = 0.42, xend = 0.975, yend = 0.215)) +
    geom_text(data = data.frame(y = exy, label = extxt), aes(x = 0.55, y = y, label = label),
              hjust = 0, size = 2.0, lineheight = 1.15, colour = INK) +
    annotate("text", x = 0.0075, y = 0.29, hjust = 0, size = 2.2, fontface = "bold",
             label = "Locked hold-out") +
    annotate("text", x = 0.544, y = 0.29, hjust = 0, size = 2.2, fontface = "bold",
             label = "Nested cross-validation") +
    coord_cartesian(xlim = c(0.005, 1.0), ylim = c(0.0, 0.99), expand = FALSE) +
    theme_blank() + labs(tag = "a") + tag_theme

  md <- median(per$events); q <- quantile(per$events, c(0.25, 0.75))
  pb <- ggplot(per, aes(events)) +
    geom_histogram(bins = 34, fill = "#9EC9E8", colour = "white", size = 0.15) +
    geom_vline(xintercept = md, colour = RED, size = 0.4) +
    annotate("text", x = md * 0.92, y = Inf, vjust = 1.5, hjust = 1, size = 2.0, colour = RED,
             label = sprintf("median %s", comma(md))) +
    scale_x_log10(breaks = c(1, 3, 10, 30, 100, 250)) +
    labs(x = "Events contributed by one child (log scale)", y = "Children", tag = "b") +
    theme_pub() + tag_theme

  (pa / pb) + plot_layout(heights = c(2.5, 1))
}

# ---------------------------------------------------------------- Fig 3: curves
fig3 <- function() {
  idx <- function(d) d %>% group_by(task) %>%
    mutate(k = as.integer(factor(class, levels = unique(class)))) %>% ungroup()
  roc <- idx(rd("curves_roc.csv") %>% mutate(task = fct(task)))
  pr  <- idx(rd("curves_pr.csv")  %>% mutate(task = fct(task)))
  rel <- rd("curves_reliability.csv") %>% mutate(task = fct(task))
  hd  <- rd("curves_headline.csv") %>% mutate(task = fct(task))
  leg <- roc %>% distinct(task, class, k) %>% arrange(task, k) %>%
    mutate(yy = 0.26 - 0.075 * k)
  lt <- c("solid", "22", "12", "4212")

  common <- list(scale_colour_manual(values = COL, guide = "none"),
                 scale_linetype_manual(values = lt, guide = "none"),
                 coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE),
                 scale_x_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1),
                                    labels = c("0", ".25", ".5", ".75", "1")),
                 scale_y_continuous(breaks = c(0, 0.25, 0.5, 0.75, 1),
                                    labels = c("0", ".25", ".5", ".75", "1")),
                 theme_pub())

  p1 <- ggplot(roc, aes(fpr, tpr, colour = task, linetype = factor(k))) +
    geom_abline(slope = 1, intercept = 0, colour = GRIDC, size = 0.3) +
    geom_line(size = 0.5) +
    geom_segment(data = leg, aes(x = 0.40, xend = 0.52, y = yy, yend = yy, colour = task,
                 linetype = factor(k)), size = 0.45, inherit.aes = FALSE) +
    geom_text(data = leg, aes(x = 0.55, y = yy, label = class), hjust = 0, size = 1.85,
              colour = INK, inherit.aes = FALSE) +
    facet_wrap(~ task, nrow = 1) + common +
    labs(x = "1 - specificity", y = "Sensitivity", tag = "a") + tag_theme

  p2 <- ggplot(pr, aes(recall, precision, colour = task, linetype = factor(k))) +
    geom_hline(aes(yintercept = prevalence, colour = task), size = 0.25, linetype = "13") +
    geom_line(size = 0.5) +
    facet_wrap(~ task, nrow = 1) + common +
    labs(x = "Sensitivity (recall)", y = "Positive predictive value", tag = "b") + tag_theme

  p3 <- ggplot(rel, aes(conf, acc, colour = task)) +
    geom_abline(slope = 1, intercept = 0, colour = GRIDC, size = 0.3) +
    geom_line(size = 0.5) + geom_point(size = 0.9) +
    facet_wrap(~ task, nrow = 1) + common +
    labs(x = "Mean predicted confidence", y = "Observed accuracy", tag = "c") + tag_theme

  (p1 / p2 / p3)
}

# ---------------------------------------------------------------- Fig 4: ladder
fig4 <- function() {
  lv <- rev(c("Event duration only", "Demographics only", "Own-task base model",
              "All acoustic probabilities", "Full stack"))
  d <- rd("ladder.csv") %>%
    mutate(task = fct(task), model = factor(model, levels = lv),
           is_full = model == "Full stack")
  band <- d %>% filter(model == "Event duration only") %>%
    transmute(task, xmax = auc) %>% mutate(xmin = 0.45)
  inside <- d %>% filter(model == "Full stack") %>% left_join(band, by = "task") %>%
    filter(lo <= xmax) %>%
    mutate(note = "full stack does not clear\nthe duration shortcut")

  ggplot(d, aes(auc, model, colour = task)) +
    geom_rect(data = band, inherit.aes = FALSE,
              aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
              fill = "#EDEDEA") +
    geom_vline(data = band, aes(xintercept = xmax), colour = MUTED, size = 0.3,
               linetype = "22") +
    geom_linerange(aes(xmin = lo, xmax = hi), size = 0.45) +
    geom_point(aes(shape = is_full), size = 1.5, fill = "white", stroke = 0.5) +
    geom_text(aes(x = hi + 0.012, label = sprintf("%.2f", auc)), hjust = 0, size = 1.9,
              colour = INK) +
    scale_shape_manual(values = c(`FALSE` = 21, `TRUE` = 19), guide = "none") +
    scale_colour_manual(values = COL, guide = "none") +
    scale_x_continuous(limits = c(0.45, 1.07), breaks = seq(0.5, 1, 0.1)) +
    facet_wrap(~ task, nrow = 1) +
    labs(x = "AUC (event level, nested cross-validation)", y = NULL) +
    theme_pub() + theme(panel.grid.major.y = element_blank())
}

# ---------------------------------------------------------------- Fig 5: leakage
fig5 <- function() {
  a <- rd("arms.csv") %>% mutate(task = fct(task),
                                 level = factor(level, c("Event level", "Patient level")))
  wide <- a %>% select(arm, task, level, auc) %>% tidyr::pivot_wider(names_from = arm,
                                                                    values_from = auc)
  off <- c(L0 = 0, L1 = 0.22, L3 = -0.22)
  pts <- a %>% filter(arm %in% c("L0", "L1", "L3")) %>%
    mutate(arm = factor(arm, c("L0", "L1", "L3")),
           yn = as.numeric(level) + off[as.character(arm)])
  wide <- wide %>% mutate(yn = as.numeric(level))
  armlab <- c(L0 = "L0  released encoder, patient-level split (reference)",
              L1 = "L1  encoder previously fine-tuned on an event-level split",
              L3 = "L3  earlier preprint (not a controlled arm)")
  meta <- rd("common_events_meta.csv")

  p1 <- ggplot() +
    geom_segment(data = wide, aes(x = L0, xend = L2, y = yn, yend = yn, colour = task),
                 size = 0.6, arrow = arrow(length = unit(1.4, "mm"), type = "closed")) +
    geom_text(data = wide, aes(x = (L0 + L2) / 2, y = yn,
              label = sprintf("+%.2f", L2 - L0)), vjust = -0.8, size = 2.0, colour = INK) +
    geom_point(data = pts, aes(auc, yn, shape = arm, colour = task), size = 1.6,
               fill = "white", stroke = 0.5) +
    scale_shape_manual(values = c(L0 = 19, L1 = 15, L3 = 23), labels = armlab) +
    scale_colour_manual(values = COL, guide = "none") +
    scale_x_continuous(limits = c(0.45, 1.0), breaks = seq(0.5, 1, 0.1)) +
    scale_y_reverse(breaks = c(1, 2), labels = levels(a$level),
                    limits = c(2.55, 0.45)) +
    facet_wrap(~ task, nrow = 1) +
    labs(x = "AUC", y = NULL, tag = "a") +
    guides(shape = guide_legend(ncol = 1, override.aes = list(colour = INK))) +
    theme_pub() +
    theme(panel.grid.major.y = element_blank(), legend.position = "bottom") + tag_theme

  ce <- rd("common_events.csv") %>% mutate(task = fct(task))
  pt2 <- ce %>% filter(arm != "delta") %>%
    mutate(arm = factor(arm, c("L0", "L2"),
                        c("L0  patient never seen", "L2  patient seen in training")))
  dl <- ce %>% filter(arm == "delta")
  p2 <- ggplot(pt2, aes(auc, arm, colour = task)) +
    geom_linerange(aes(xmin = lo, xmax = hi), size = 0.45) +
    geom_point(size = 1.6) +
    geom_text(data = dl, aes(x = 0.455, y = 1.52,
              label = sprintf("%+.3f (%.3f to %.3f)", auc, lo, hi)),
              hjust = 0, size = 2.0, colour = INK) +
    scale_colour_manual(values = COL, guide = "none") +
    scale_y_discrete(limits = rev) +
    scale_x_continuous(limits = c(0.45, 1.02), breaks = seq(0.5, 1, 0.1)) +
    facet_wrap(~ task, nrow = 1) +
    labs(y = NULL, tag = "b",
         x = sprintf("AUC on the %s events held out by both arms", comma(meta$n_events))) +
    theme_pub() + theme(panel.grid.major.y = element_blank()) + tag_theme

  (p1 / p2) + plot_layout(heights = c(1.5, 1))
}

# ---------------------------------------------------------------- Fig 6: attribution
fig6 <- function() {
  mp <- rd("saliency_maps.csv") %>% mutate(task = fct(task))
  pf <- rd("saliency_profile.csv") %>% mutate(task = fct(task))
  dl <- rd("saliency_deletion.csv") %>% mutate(task = fct(task))
  bd <- rd("mel_bands.csv")
  cut_band <- max(bd$band[bd$high_hz <= 1800]) + 0.5
  brk <- c(1, 3, 5, 7); blab <- sprintf("%.1f", bd$low_hz[brk] / 1000)
  mp <- mp %>% mutate(panel = factor(paste0(task, "\n", class),
                                     levels = unique(paste0(task, "\n", class))))

  p1 <- ggplot(mp, aes(time, band, fill = value)) +
    geom_raster() +
    geom_hline(yintercept = cut_band, colour = RED, size = 0.35, linetype = "22") +
    scale_fill_gradient(low = "white", high = "#08306B", limits = c(0, 1),
                        name = "Mean probability drop\n(relative to panel maximum)") +
    scale_y_continuous(breaks = brk, labels = blab, expand = c(0, 0)) +
    scale_x_continuous(breaks = c(0, 1, 2), labels = c("0", "1", "2"),
                       expand = c(0, 0)) +
    facet_wrap(~ panel, nrow = 1) +
    labs(x = "Time in clip (s)", y = "Mel band, lower edge (kHz)", tag = "a") +
    theme_pub() +
    theme(panel.grid = element_blank(), legend.position = "right",
          legend.key.width = unit(2.6, "mm"), legend.key.height = unit(7, "mm"),
          legend.title = element_text(size = 5.6)) +
    tag_theme

  p2 <- ggplot(pf, aes(band, share, colour = task, linetype = class)) +
    geom_vline(xintercept = cut_band, colour = RED, size = 0.35, linetype = "22") +
    geom_line(size = 0.5) + geom_point(size = 0.9) +
    scale_colour_manual(values = COL, guide = "none") +
    scale_linetype_manual(values = c("solid", "22", "11", "4212")) +
    scale_x_continuous(breaks = 1:8) +
    labs(x = "Mel band (1 = lowest frequency)",
         y = "Share of positive attribution", tag = "b") +
    theme_pub() + theme(legend.position = "right") + tag_theme

  p3 <- ggplot(dl, aes(median, task, colour = task, shape = order)) +
    geom_linerange(aes(xmin = lo, xmax = hi),
                   position = position_dodge(width = 0.45), size = 0.45) +
    geom_point(position = position_dodge(width = 0.45), size = 1.6, fill = "white",
               stroke = 0.5) +
    scale_shape_manual(values = c(`Attribution-ordered` = 19, `Random order` = 21)) +
    scale_colour_manual(values = COL, guide = "none") +
    scale_x_continuous(limits = c(0, 1)) +
    scale_y_discrete(limits = rev) +
    labs(x = "Area under the deletion curve (median, IQR; lower = more faithful)",
         y = NULL, tag = "c") +
    guides(shape = guide_legend(override.aes = list(colour = INK))) +
    theme_pub() + theme(panel.grid.major.y = element_blank(),
                        legend.position = "right") + tag_theme

  (p1 / p2 / p3) + plot_layout(heights = c(1.15, 1, 0.72))
}

FIGS <- list(`1` = list(fig1, 46), `2` = list(fig2, 122), `3` = list(fig3, 168),
             `4` = list(fig4, 56), `5` = list(fig5, 120), `6` = list(fig6, 150))
for (n in ONLY) {
  f <- FIGS[[n]]
  save_fig(f[[1]](), paste0("Fig", n), f[[2]])
}
