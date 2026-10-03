"""本地 Tkinter 交互窗口：图片/随机运动、实时预览、参数调整和后台导出。"""
from __future__ import annotations

import copy
from contextlib import redirect_stderr, redirect_stdout
import math
from pathlib import Path
import queue
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2

from main import ROOT, Renderer, build_parser, encode_video, make_renderer


class FourierApp:
    BG = "#090d16"
    PANEL = "#141b29"
    TEXT = "#eef2f8"
    MUTED = "#a4b1c5"
    ACCENT = "#77d9e5"
    PREVIEW_BUILD_TIMEOUT = 30

    def __init__(self, root, defaults=None):
        self.root = root
        self.defaults = defaults or build_parser().parse_args(["--headless"])
        self.events = queue.SimpleQueue()
        self.controls = None
        self.renderer = None
        self.active_settings = None
        self.build_id = 0
        self.building = False
        self.build_started = 0.0
        self.exporting = False
        self.closed = False
        self.export_thread = None
        self.export_resume_playing = False
        self.export_started = 0.0
        self.export_last_progress = 0.0
        self.cancel_event = threading.Event()
        self.playing = False
        self.scene_seconds = 0.0
        self.last_clock = time.perf_counter()
        self.preview_frames = 0
        self.preview_pending = False
        self.preview_failed = False
        self.preview_requests = queue.Queue(maxsize=1)
        self.preview_stop = threading.Event()
        self.preview_thread = threading.Thread(target=self._render_preview,
                                               args=(self.preview_requests, self.events, self.preview_stop),
                                               daemon=True)
        self.preview_thread.start()
        self.last_frame = None
        self.photo = None
        self.status = tk.StringVar(root, "选择输入、调节参数，然后开始预览。")
        self.progress = tk.DoubleVar(root, 0)
        self.values = {}
        self.source = tk.StringVar(root, getattr(self.defaults, "source", "text"))
        self.mode = tk.StringVar(root, "drift")
        self.color = tk.StringVar(root, "cyan")
        self.speed = tk.DoubleVar(root, 1.0)
        self._style()
        self._launcher()
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)
        self.poll_id = self.root.after(20, self._poll)

    def _style(self):
        self.root.configure(bg=self.BG)
        self.root.option_add("*Font", ("Microsoft YaHei UI", 10))
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=self.PANEL, foreground=self.TEXT, font=("Microsoft YaHei UI", 10))
        style.configure("TFrame", background=self.PANEL)
        style.configure("TLabel", background=self.PANEL, foreground=self.TEXT)
        style.configure("Muted.TLabel", foreground=self.MUTED)
        style.configure("TButton", padding=(12, 8), background="#263347", foreground=self.TEXT)
        style.map("TButton", background=[("active", "#34455d"), ("disabled", "#1c2433")],
                  foreground=[("disabled", "#62718a")])
        style.configure("Accent.TButton", background=self.ACCENT, foreground="#091522")
        style.map("Accent.TButton", background=[("active", "#a5ecf3"), ("disabled", "#304b59")])
        style.configure("TEntry", fieldbackground="#0c1320", foreground=self.TEXT, insertcolor=self.TEXT)
        style.configure("TCombobox", fieldbackground="#0c1320", foreground=self.TEXT)
        style.map("TCombobox", fieldbackground=[("readonly", "#0c1320")],
                  foreground=[("readonly", self.TEXT)])
        style.configure("TRadiobutton", background=self.PANEL, foreground=self.TEXT)
        style.configure("Horizontal.TProgressbar", troughcolor="#0c1320", background=self.ACCENT)
        self.root.option_add("*TCombobox*Listbox.background", self.PANEL)
        self.root.option_add("*TCombobox*Listbox.foreground", self.TEXT)

    def _launcher(self):
        self.root.title("Fourier Studio · 动态数学壁纸")
        self.root.geometry("660x470")
        self.root.minsize(600, 430)
        tk.Label(self.root, text="FOURIER STUDIO", bg=self.BG, fg=self.ACCENT,
                 font=("Segoe UI", 12, "bold")).pack(pady=(45, 8))
        tk.Label(self.root, text="让数学缓慢生长", bg=self.BG, fg=self.TEXT,
                 font=("Microsoft YaHei UI", 26, "bold")).pack()
        tk.Label(self.root, text="旋转圆盘 · 漂移曲线 · 渐隐尾迹", bg=self.BG,
                 fg=self.MUTED).pack(pady=(12, 20))
        decoration = tk.Canvas(self.root, width=400, height=100, bg=self.BG, highlightthickness=0)
        decoration.pack()
        for x, y, radius, color in [(145, 48, 40, "#df806e"), (225, 60, 27, "#98b894"),
                                     (273, 40, 18, "#bf8dbc")]:
            decoration.create_oval(x-radius, y-radius, x+radius, y+radius, outline=color, width=2)
            decoration.create_line(x, y, x+radius*.75, y-radius*.55, fill="#e9eef4", width=2, arrow="last")
        self.launch_button = ttk.Button(self.root, text="开始运行", style="Accent.TButton",
                                        command=self.open_controls)
        self.launch_button.pack(pady=(12, 14), ipadx=45)
        tk.Label(self.root, text="选择本地图片，或探索可复现的随机运动", bg=self.BG, fg=self.MUTED).pack()

    def _value(self, name, default):
        if name not in self.values:
            self.values[name] = tk.StringVar(self.root, str(default))
        return self.values[name]

    def _entry(self, parent, label, name, default):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label).pack(side="left")
        entry = ttk.Entry(row, textvariable=self._value(name, default), width=12)
        entry.pack(side="right")
        return entry

    def _section(self, parent, title):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", padx=15, pady=(13, 4))
        ttk.Label(frame, text=title, foreground=self.ACCENT,
                  font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
        return frame

    def _slider(self, parent, label, name, default, start, end, step):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(6, 0))
        tk.Scale(parent, variable=self._value(name, default), from_=start, to=end,
                 resolution=step, orient="horizontal", bg=self.PANEL, fg=self.TEXT,
                 troughcolor="#253247", activebackground=self.ACCENT, highlightthickness=0,
                 relief="flat", borderwidth=0, length=260).pack(fill="x")

    def open_controls(self):
        if self.controls is not None:
            self.controls.lift()
            return
        self.controls = tk.Toplevel(self.root)
        self.controls.title("Fourier Studio · 参数与预览")
        self.controls.geometry("1260x850")
        self.controls.minsize(980, 700)
        self.controls.configure(bg=self.BG)
        self.controls.protocol("WM_DELETE_WINDOW", self.close_controls)
        self.root.withdraw()
        self.controls.columnconfigure(1, weight=1)
        self.controls.rowconfigure(1, weight=1)
        header = tk.Frame(self.controls, bg=self.BG)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=22, pady=18)
        tk.Label(header, text="FOURIER STUDIO", bg=self.BG, fg=self.TEXT,
                 font=("Segoe UI", 17, "bold")).pack(side="left")
        tk.Label(header, text="生成艺术 / 平滑循环", bg=self.BG, fg=self.MUTED).pack(side="right")
        side = ttk.Frame(self.controls)
        side.grid(row=1, column=0, sticky="nsew", padx=(18, 10), pady=(0, 12))
        side.rowconfigure(0, weight=1)
        scroll = tk.Canvas(side, width=320, bg=self.PANEL, highlightthickness=0)
        scroll.grid(row=0, column=0, sticky="ns")
        scrollbar = ttk.Scrollbar(side, orient="vertical", command=scroll.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        scroll.configure(yscrollcommand=scrollbar.set)
        body = ttk.Frame(scroll)
        scroll.create_window((0, 0), window=body, anchor="nw", width=320)
        body.bind("<Configure>", lambda event: scroll.configure(scrollregion=scroll.bbox("all")))
        self.sidebar = scroll
        self.controls.bind("<MouseWheel>", self._scroll)
        source_section = self._section(body, "01  选择运动来源")
        for label, value in [("上传本地图片", "image"), ("输入文字", "text"), ("随机运动", "random")]:
            ttk.Radiobutton(source_section, text=label, value=value, variable=self.source,
                            command=self._source_changed).pack(anchor="w", pady=4)
        self.image_group = ttk.Frame(source_section)
        self._value("input", self.defaults.input)
        ttk.Entry(self.image_group, textvariable=self.values["input"], width=32).pack(fill="x", pady=5)
        self.browse_button = ttk.Button(self.image_group, text="选择 SVG / PNG / JPG", command=self.choose_image)
        self.browse_button.pack(fill="x")
        ttk.Label(self.image_group, text="图片仅在本机读取，不上传到网络。",
                  style="Muted.TLabel").pack(anchor="w", pady=5)

        self.text_group = ttk.Frame(source_section)
        self._value("text", getattr(self.defaults, "text", "傅里叶"))
        ttk.Label(self.text_group, text="输入名字 / 汉字：").pack(anchor="w", pady=(2, 2))
        ttk.Entry(self.text_group, textvariable=self.values["text"], width=32).pack(fill="x", pady=(0, 6))

        ttk.Label(self.text_group, text="字形来源：").pack(anchor="w", pady=(2, 2))
        styles = {"ordered": "ordered (标准笔顺单线·推荐)",
                  "signature": "signature (行书字体映射·实验)",
                  "contour": "contour (字体外轮廓)"}
        style = getattr(self.defaults, "text_style", "ordered")
        self._value("text_style", styles.get(style, style))
        self.style_combo = ttk.Combobox(self.text_group, textvariable=self.values["text_style"],
                                       values=list(styles.values()), state="readonly")
        self.style_combo.pack(fill="x", pady=(0, 6))
        self.style_combo.bind("<<ComboboxSelected>>", self._text_style_changed)

        self._value("font", getattr(self.defaults, "font", "C:/Windows/Fonts/STXINGKA.TTF"))
        from signature import FONT_PRESETS
        self.available_fonts={name:path for name,path in FONT_PRESETS.items() if path.is_file()}
        current=Path(self.values["font"].get())
        name=next((name for name,path in self.available_fonts.items() if path==current),"自选字体")
        self.font_preset=tk.StringVar(self.root,name)
        ttk.Label(self.text_group,text="字体（实验映射 / 外轮廓）").pack(anchor="w",pady=(5,2))
        self.font_combo=ttk.Combobox(self.text_group,textvariable=self.font_preset,
                                   values=[*self.available_fonts,"自选字体"],state="readonly")
        self.font_combo.pack(fill="x",pady=(0,4))
        self.font_combo.bind("<<ComboboxSelected>>",self._font_changed)
        font_row = ttk.Frame(self.text_group)
        font_row.pack(fill="x", pady=2)
        ttk.Label(font_row, text="本地字体（实验映射 / 外轮廓模式）：").pack(side="left")
        self.font_browse_button = ttk.Button(font_row, text="浏览字体", command=self.choose_font)
        self.font_browse_button.pack(side="right")
        self.font_entry = ttk.Entry(self.text_group, textvariable=self.values["font"], width=32)
        self.font_entry.pack(fill="x", pady=(2, 4))
        self.close_loop_var = tk.BooleanVar(self.root, True)
        ttk.Checkbutton(self.text_group, text="首尾相连（曲线从字外绕回起点）",
                        variable=self.close_loop_var).pack(anchor="w", pady=4)
        self.show_transfers_var = tk.BooleanVar(self.root, getattr(self.defaults, "show_transfers", False))
        ttk.Checkbutton(self.text_group, text="显示全部连接线（含字内换笔）",
                        variable=self.show_transfers_var).pack(anchor="w", pady=4)
        self.text_help = ttk.Label(self.text_group, foreground=self.ACCENT)
        self.text_help.pack(anchor="w", pady=2)
        self._text_style_changed()

        self.random_group = ttk.Frame(source_section)
        self._entry(self.random_group, "随机种子", "seed", self.defaults.seed)
        self._entry(self.random_group, "最高频率", "max_frequency", self.defaults.max_frequency)
        self._entry(self.random_group, "高频衰减", "spectral_decay", self.defaults.spectral_decay)
        self.regenerate_button = ttk.Button(self.random_group, text="换一组随机运动", command=self.regenerate)
        self.regenerate_button.pack(fill="x", pady=5)
        self._source_changed()
        motion = self._section(body, "02  运动与尾迹")
        ttk.Combobox(motion, textvariable=self.mode, values=["drift", "loop", "draw"],
                     state="readonly").pack(fill="x")
        ttk.Label(motion, text="drift 漂移渐隐 / loop 白色闭合曲线\ndraw 从零绘制",
                  style="Muted.TLabel").pack(anchor="w", pady=5)
        self._entry(motion, "完整循环（秒）", "duration", 60)
        self._entry(motion, "基础旋转周期（秒）", "rotation_period", 10)
        self._entry(motion, "圆盘数量", "circles", 180 if self.source.get() == "text" else 60)
        self._entry(motion, "轮廓采样数", "samples", 2000)
        self._slider(motion, "漂移强度（笔顺模式采用固定系数）", "drift_strength", .18, 0, .75, .01)
        self._entry(motion, "漂移周期（秒）", "drift_period", 30)
        self._entry(motion, "尾迹保留（秒）", "trail_length", 8)
        self._slider(motion, "渐隐速度", "fade_power", 2, .2, 8, .1)
        self._slider(motion, "发光强度", "glow_strength", .8, 0, 3, .05)
        ttk.Label(motion, text="尾迹颜色").pack(anchor="w", pady=(8, 4))
        ttk.Combobox(motion, textvariable=self.color, values=["cyan", "blue", "purple"],
                     state="readonly").pack(fill="x")
        style = self._section(body, "03  圆盘样式")
        self._entry(style, "圆环线宽", "ring_width", 2.4)
        self._entry(style, "圆环亮度", "ring_opacity", .90)
        self._entry(style, "刻度数量", "ticks", 12)
        output = self._section(body, "04  视频输出")
        self.resolution = tk.StringVar(self.root, "1920x1080")
        ttk.Combobox(output, textvariable=self.resolution,
                     values=["1920x1080", "3840x2160", "1280x720", "960x540"],
                     state="readonly").pack(fill="x")
        self._entry(output, "导出 FPS", "fps", 60)
        ttk.Label(output, text="预览自适应帧率，导出使用此帧率。",
                  style="Muted.TLabel").pack(anchor="w", pady=6)
        actions = ttk.Frame(side)
        actions.grid(row=1, column=0, columnspan=2, sticky="ew", padx=12, pady=10)
        self.apply_button = ttk.Button(actions, text="开始预览 / 应用参数", style="Accent.TButton",
                                      command=self.prepare_preview)
        self.apply_button.pack(fill="x")
        ttk.Label(actions, text="修改参数后点击应用；左栏可滚动。",
                  style="Muted.TLabel").pack(pady=(6, 0))
        right = tk.Frame(self.controls, bg=self.BG)
        right.grid(row=1, column=1, sticky="nsew", padx=(0, 18), pady=(0, 12))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(right, bg="black", highlightthickness=1, highlightbackground="#263245")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas_item = self.canvas.create_image(0, 0, anchor="center")
        self.canvas.create_text(380, 240, text="选择来源，然后点击开始预览", fill=self.MUTED,
                                font=("Microsoft YaHei UI", 13), tags="placeholder")
        toolbar = ttk.Frame(right)
        toolbar.grid(row=1, column=0, sticky="ew", pady=10)
        self.pause_button = ttk.Button(toolbar, text="暂停", command=self.toggle_pause, state="disabled")
        self.pause_button.pack(side="left")
        ttk.Label(toolbar, text="预览速度").pack(side="left", padx=(14, 5))
        tk.Scale(toolbar, variable=self.speed, from_=.25, to=2, resolution=.05,
                 orient="horizontal", length=160, bg=self.PANEL, fg=self.TEXT,
                 troughcolor="#253247", highlightthickness=0).pack(side="left")
        self.export_button = ttk.Button(toolbar, text="导出当前预览 MP4", command=self.export_video,
                                       style="Accent.TButton", state="disabled")
        self.export_button.pack(side="right")
        self.cancel_button = ttk.Button(toolbar, text="取消导出", command=self.cancel_export, state="disabled")
        self.cancel_button.pack(side="right", padx=7)
        ttk.Progressbar(right, variable=self.progress, maximum=100).grid(row=2, column=0, sticky="ew")
        ttk.Label(right, textvariable=self.status, style="Muted.TLabel", wraplength=780).grid(
            row=3, column=0, sticky="ew", pady=(8, 0))

    def _scroll(self, event):
        if self.controls is not None and event.x_root < self.sidebar.winfo_rootx() + self.sidebar.winfo_width():
            self.sidebar.yview_scroll(-int(event.delta / 120), "units")

    def _source_changed(self):
        self.image_group.pack_forget()
        self.text_group.pack_forget()
        self.random_group.pack_forget()
        src = self.source.get()
        if src == "image":
            self.image_group.pack(fill="x", pady=5)
        elif src == "text":
            self.text_group.pack(fill="x", pady=5)
        else:
            self.random_group.pack(fill="x", pady=5)

    def choose_image(self):
        file = filedialog.askopenfilename(parent=self.controls, title="选择图形",
                                         filetypes=[("图形", "*.svg *.png *.jpg *.jpeg *.bmp *.webp"),
                                                    ("所有文件", "*.*")])
        if file:
            self.values["input"].set(file)
            self.status.set("已选择图片。点击开始预览 / 应用参数。")

    def _text_style_changed(self, event=None):
        ordered = self.values["text_style"].get().split()[0]=="ordered"
        self.font_combo.configure(state="disabled" if ordered else "readonly")
        for control in (self.font_entry, self.font_browse_button):
            control.configure(state="disabled" if ordered else "normal")
        self.text_help.configure(text=(
            "标准笔顺数据直接生成单线字形，不使用本地字体。\n换笔不留墨；各盘匀速、线宽固定，盘数决定细节。"
            if ordered else
            "字体映射属于实验功能，不能保证任意字的字形和笔顺。\n外轮廓模式保留字体边界，不代表手写笔顺。"))

    def _font_changed(self,event=None):
        choice=self.font_preset.get()
        if choice in self.available_fonts:
            self.values["font"].set(str(self.available_fonts[choice]))
            self.status.set(f"已选择{choice}。点击开始预览 / 应用参数。")
        else:
            self.choose_font()

    def choose_font(self):
        file = filedialog.askopenfilename(
            parent=self.controls, title="选择字体文件",
            initialdir="C:/Windows/Fonts",
            filetypes=[("字体文件", "*.ttf *.ttc *.otf"), ("所有文件", "*.*")]
        )
        if file:
            self.values["font"].set(file)
            self.font_preset.set("自选字体")
            self.status.set("已选择字体。点击开始预览 / 应用参数。")

    def read_settings(self):
        settings = copy.copy(self.defaults)
        settings.source, settings.trail_mode, settings.trail_color = self.source.get(), self.mode.get(), self.color.get()
        settings.input = Path(self.values["input"].get())
        settings.text = self.values.get("text", tk.StringVar(self.root, "傅里叶")).get()
        raw_style = self.values.get("text_style", tk.StringVar(self.root, "ordered")).get()
        settings.text_style = raw_style.split()[0] if " " in raw_style else raw_style
        settings.font = self.values.get("font", tk.StringVar(self.root, "C:/Windows/Fonts/STXINGKA.TTF")).get()
        settings.close_loop = getattr(self, "close_loop_var", tk.BooleanVar(self.root, True)).get()
        settings.show_transfers = self.show_transfers_var.get()
        integer_names = {"circles", "samples", "seed", "max_frequency", "ticks", "fps"}
        for name, variable in self.values.items():
            if name in {"input", "text", "font", "text_style"}:
                continue
            setattr(settings, name, int(variable.get()) if name in integer_names else float(variable.get()))
        settings.width, settings.height = [int(v) for v in self.resolution.get().split("x")]
        floats = ["duration", "rotation_period", "drift_strength", "drift_period", "trail_length",
                  "fade_power", "glow_strength", "ring_width", "ring_opacity", "spectral_decay"]
        if any(not math.isfinite(getattr(settings, name)) for name in floats):
            raise ValueError("所有参数必须为有限数值。")
        if settings.duration <= 0 or settings.fps <= 0 or round(settings.duration * settings.fps) < 2:
            raise ValueError("时长和 FPS 必须大于零，并至少产生两帧。")
        if settings.samples < 16 or not 1 <= settings.circles <= settings.samples:
            raise ValueError("采样数至少 16，圆盘数为 1～采样数。")
        if settings.seed < 0 or settings.max_frequency < 1 or not .3 <= settings.spectral_decay <= 3:
            raise ValueError("种子不能为负；最高频率至少 1；高频衰减为 0.3～3。")
        if not 0 < settings.ring_width <= 12 or not 0 <= settings.ring_opacity <= 1 or not 0 <= settings.ticks <= 60:
            raise ValueError("线宽为 0～12（不含 0）、亮度为 0～1、刻度为 0～60。")
        return settings

    def regenerate(self):
        if self.exporting:
            return
        if self.building:
            return
        self.source.set("random")
        self._source_changed()
        try:
            self.values["seed"].set(int(self.values["seed"].get()) + 1)
        except ValueError:
            self.values["seed"].set(42)
        self.prepare_preview()

    def prepare_preview(self):
        if self.building or self.exporting or self.closed or self.controls is None:
            return
        try:
            settings = self.read_settings()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("参数错误", str(exc), parent=self.controls)
            return
        self.building = True
        self.build_started = time.perf_counter()
        self.playing = False
        self.build_id += 1
        build_id = self.build_id
        self.apply_button.configure(state="disabled")
        self.status.set("正在准备轮廓、周期漂移和上一周期尾迹…")
        events = self.events
        def worker():
            try:
                renderer = make_renderer(settings, 800, 450)
                events.put(("ready", build_id, renderer, settings))
            except (Exception, SystemExit) as exc:
                traceback.print_exc()
                events.put(("build_error", build_id, str(exc)))
        threading.Thread(target=worker, daemon=True).start()

    def toggle_pause(self):
        self.playing = not self.playing
        self.last_clock = time.perf_counter()
        self.pause_button.configure(text="暂停" if self.playing else "继续")

    def export_video(self, output=None):
        if self.exporting or self.renderer is None:
            return
        if output is None:
            output = filedialog.asksaveasfilename(parent=self.controls, title="导出 MP4",
                                                 initialfile="Fourier_Wallpaper.mp4",
                                                 defaultextension=".mp4",
                                                 filetypes=[("MP4 视频", "*.mp4")])
        if not output:
            return
        output = Path(output)
        settings = copy.copy(self.active_settings)
        if settings.source == "image" and output.resolve() == settings.input.resolve():
            messagebox.showerror("输出错误", "输出不能覆盖输入图片。", parent=self.controls)
            return
        preview_renderer = self.renderer
        self.export_resume_playing = self.playing
        self.playing = False
        self.exporting = True
        self.export_started = self.export_last_progress = time.perf_counter()
        self.pause_button.configure(state="disabled", text="导出中暂停")
        self.apply_button.configure(state="disabled")
        self.cancel_event = threading.Event()
        cancel = self.cancel_event
        events, export_started = self.events, self.export_started
        self.export_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress.set(0)
        self.status.set("正在准备导出；已暂停预览以集中计算资源…")
        def worker():
            try:
                renderer = Renderer(preview_renderer.model, settings.width, settings.height, settings.trail_mode,
                                    ring_width=settings.ring_width, ring_opacity=settings.ring_opacity,
                                    ticks=settings.ticks, fading=preview_renderer.fading, render_fps=settings.fps)
                def progress(done, total):
                    elapsed = time.perf_counter() - export_started
                    rate = done / max(elapsed, .001)
                    remaining = (total-done) / rate
                    events.put(("progress", done, total, elapsed, rate, remaining))
                encode_video(renderer, output, settings.fps, settings.duration, settings.crf, settings.preset,
                             settings.encoder, settings.ffmpeg, True, progress=progress, cancel_event=cancel)
                events.put(("export_done", str(output.resolve())))
            except Exception as exc:
                events.put(("export_error", "导出已取消。" if cancel.is_set() else str(exc)))
        self.export_thread = threading.Thread(target=worker, daemon=False)
        self.export_thread.start()

    def cancel_export(self):
        self.cancel_event.set()
        self.status.set("正在取消导出并清理临时视频…")

    @staticmethod
    def _render_preview(requests, events, stop):
        """Never retain the app or Tk objects in a background thread."""
        while not stop.is_set():
            try:
                build_id, renderer, phase = requests.get(timeout=.1)
            except queue.Empty:
                continue
            try:
                frame = renderer.frame(phase)
                events.put(("preview_frame", build_id, frame))
            except Exception as exc:
                traceback.print_exc()
                events.put(("preview_error", build_id, str(exc)))

    def _preview_error(self, error):
        self.playing = False
        self.preview_failed = True
        self.status.set("预览失败：" + str(error))
        self.pause_button.configure(text="继续")

    def _display(self, frame):
        width, height = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        scale = min(width / frame.shape[1], height / frame.shape[0])
        target = (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale)))
        fitted = cv2.resize(frame, target, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(fitted, cv2.COLOR_BGR2RGB)
        data = f"P6 {target[0]} {target[1]} 255\n".encode("ascii") + rgb.tobytes()
        self.photo = tk.PhotoImage(master=self.controls, data=data, format="PPM")
        self.canvas.itemconfigure(self.canvas_item, image=self.photo)
        self.canvas.coords(self.canvas_item, width / 2, height / 2)
        self.canvas.delete("placeholder")

    def _poll(self):
        if self.closed:
            return
        started = time.perf_counter()
        while not self.events.empty():
            event = self.events.get()
            kind = event[0]
            if kind == "ready" and event[1] == self.build_id and self.controls is not None:
                self.renderer, self.active_settings = event[2], event[3]
                self.building, self.playing = False, not self.exporting
                self.scene_seconds = 0
                self.last_frame = None
                self.preview_failed = False
                self.last_clock = time.perf_counter()
                self.apply_button.configure(state="disabled" if self.exporting else "normal")
                self.pause_button.configure(state="disabled" if self.exporting else "normal",
                                            text="导出中暂停" if self.exporting else "暂停")
                self.export_button.configure(state="disabled" if self.exporting else "normal")
                if not self.exporting:
                    if hasattr(self.renderer.model, "path"):
                        model = self.renderer.model
                        label = "标准笔顺单线" if model.path.source_kind=="ordered" else "实验字体映射"
                        self.status.set(f"正在预览 · {label} · 各盘匀速 · {len(model.radii)} 个圆盘")
                    elif self.renderer.fading is not None:
                        model = self.renderer.model
                        periods = " / ".join(f"{v:.2g}" for v in model.actual_periods)
                        self.status.set(f"正在预览 · {len(model.radii)} 个圆盘 · 完整循环 {model.loop_seconds:g}s · 漂移周期 {periods}s")
                    else:
                        self.status.set("正在预览原有白色曲线模式。修改参数后点击应用。")
            elif kind in {"preview_frame", "preview_error"}:
                self.preview_pending = False
                if event[1] == self.build_id and self.controls is not None and not self.exporting:
                    if kind == "preview_error":
                        self._preview_error(event[2])
                    else:
                        try:
                            self._display(event[2])
                            self.last_frame = event[2]
                            self.preview_frames += 1
                        except Exception as exc:
                            self._preview_error(exc)
            elif kind == "build_error" and event[1] == self.build_id and self.controls is not None:
                self.building = False
                self.apply_button.configure(state="normal")
                self.status.set("准备失败：" + event[2])
                messagebox.showerror("无法生成预览", event[2], parent=self.controls)
            elif kind == "progress" and self.controls is not None:
                _, done, total, elapsed, rate, remaining = event
                self.export_last_progress = time.perf_counter()
                self.progress.set(min(99.9, done / total * 100))
                if done == total:
                    self.status.set("所有视频帧已生成，正在收尾封装 MP4…")
                else:
                    used = f"{int(elapsed)//60:02d}:{int(elapsed)%60:02d}"
                    left = f"{math.ceil(remaining)//60:02d}:{math.ceil(remaining)%60:02d}"
                    self.status.set(f"导出 {done}/{total} 帧 · {rate:.1f} 帧/秒 · 已用 {used} · 预计剩余 {left}")
            elif kind in {"export_done", "export_error"}:
                self.exporting = False
                self.playing = self.export_resume_playing if self.controls is not None else False
                self.last_clock = time.perf_counter()
                if self.controls is not None:
                    self.pause_button.configure(state="normal", text="暂停" if self.playing else "继续")
                    self.apply_button.configure(state="normal")
                    self.export_button.configure(state="normal" if self.renderer is not None else "disabled")
                    self.cancel_button.configure(state="disabled")
                    if kind == "export_done":
                        self.progress.set(100)
                        self.status.set("导出完成：" + event[1])
                    else:
                        self.status.set(event[1])
        now = time.perf_counter()
        if self.building and self.controls is not None:
            waiting = now - self.build_started
            if waiting >= self.PREVIEW_BUILD_TIMEOUT:
                self.build_id += 1  # Ignore late results from the timed-out worker.
                self.building = False
                self.apply_button.configure(state="normal")
                self.status.set("准备超时，请重新点击开始预览；详情见项目目录 gui.log。")
                print("Preview preparation timed out after", round(waiting, 1), "seconds.")
            elif waiting >= 1:
                self.status.set(f"正在准备轮廓与尾迹… 已等待 {int(waiting)} 秒")
        if self.exporting and self.controls is not None and now-self.export_last_progress >= 10:
            waiting = int(now-self.export_last_progress)
            if self.cancel_event.is_set():
                self.status.set("正在取消导出并清理临时视频…")
            else:
                self.status.set(f"等待当前渲染/编码步骤：{waiting} 秒；超过 120 秒无进展将自动报错。")
        if self.controls is not None and self.renderer is not None:
            if self.playing:
                self.scene_seconds += (now - self.last_clock) * self.speed.get()
            self.last_clock = now
            if (not self.exporting and not self.building and not self.preview_pending
                    and not self.preview_failed and (self.playing or self.last_frame is None)):
                duration = round(self.active_settings.duration * self.active_settings.fps) / self.active_settings.fps
                self.preview_pending = True
                self.preview_requests.put_nowait((self.build_id, self.renderer,
                                                 (self.scene_seconds / duration) % 1))
        delay = max(1, round(33 - (time.perf_counter() - started) * 1000))
        self.poll_id = self.root.after(delay, self._poll)

    def close_controls(self):
        self.build_id += 1
        self.building, self.playing = False, False
        self.cancel_event.set()
        if self.controls is not None:
            self.controls.destroy()
        self.controls, self.renderer, self.last_frame = None, None, None
        self.root.deiconify()

    def shutdown(self):
        self.closed = True
        self.preview_stop.set()
        self.cancel_event.set()
        try:
            self.root.after_cancel(self.poll_id)
            self.root.destroy()
        except tk.TclError:
            pass
        self.preview_thread.join(timeout=.2)


def run_gui(defaults=None):
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise SystemExit(f"无法打开桌面窗口：{exc}。可用 --headless 直接导出视频。") from exc
    # A selected Windows console can block writes; GUI diagnostics go to disk.
    with (ROOT / "gui.log").open("a", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "Fourier Studio GUI started")
            app = FourierApp(root, defaults)
            try:
                root.mainloop()
            finally:
                app.preview_stop.set()
                app.cancel_event.set()
                app.preview_thread.join(timeout=.2)
                if app.export_thread is not None:
                    app.export_thread.join(timeout=10)
    return 0
