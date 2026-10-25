import cv2
import os
import sys
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, Manager, cpu_count
from collections import defaultdict
import html
import signal

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Timeout handler for stuck video checks
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

# Use alarm for timeout (Unix only, Windows needs different approach)
TIMEOUT_SECONDS = 5

def check_video(video_path):
    """检查单个视频文件是否损坏

    返回: (video_path, is_valid, error_message, file_size)
    """
    try:
        file_size = os.path.getsize(video_path)

        # Quick file size check - if too small, likely corrupted
        if file_size < 1024:  # Less than 1KB
            return (str(video_path), False, f"文件太小 ({file_size} bytes)", file_size)

        cap = cv2.VideoCapture(str(video_path))

        # 检查是否成功打开
        if not cap.isOpened():
            cap.release()
            return (str(video_path), False, "无法打开文件", file_size)

        # 检查基本属性
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 检查属性是否合理
        if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
            cap.release()
            return (str(video_path), False, f"异常属性: frames={frame_count}, fps={fps}, size={width}x{height}", file_size)

        # 尝试读取第一帧
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return (str(video_path), False, "无法读取第一帧", file_size)

        cap.release()
        return (str(video_path), True, "正常", file_size)

    except Exception as e:
        file_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
        return (str(video_path), False, f"异常: {str(e)}", file_size)

def check_video_wrapper(args):
    """多进程包装函数"""
    video_path, counter, lock = args
    result = check_video(video_path)

    # 更新进度
    with lock:
        counter.value += 1
        if counter.value % 500 == 0:
            # Force flush output
            print(f"已检查: {counter.value} 个文件...", flush=True)

    return result

def generate_html_report(corrupted_by_dir, total_files, corrupted_count, root_dir, elapsed_time):
    """生成 HTML 报告"""

    html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>视频完整性检查报告</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #f5f5f5;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
        }}
        .header h1 {{
            font-size: 28px;
            margin-bottom: 10px;
        }}
        .header p {{
            opacity: 0.9;
            font-size: 14px;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
            border-bottom: 1px solid #e0e0e0;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .stat-label {{
            color: #666;
            font-size: 13px;
            margin-bottom: 8px;
        }}
        .stat-value {{
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }}
        .stat-value.error {{
            color: #e53e3e;
        }}
        .stat-value.success {{
            color: #38a169;
        }}
        .search-box {{
            padding: 20px 30px;
            background: white;
            border-bottom: 1px solid #e0e0e0;
        }}
        .search-box input {{
            width: 100%;
            padding: 12px 20px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            font-size: 14px;
            transition: border-color 0.3s;
        }}
        .search-box input:focus {{
            outline: none;
            border-color: #667eea;
        }}
        .content {{
            padding: 30px;
        }}
        .directory {{
            margin-bottom: 30px;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            overflow: hidden;
        }}
        .dir-header {{
            background: #f8f9fa;
            padding: 15px 20px;
            cursor: pointer;
            user-select: none;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: background 0.2s;
        }}
        .dir-header:hover {{
            background: #e9ecef;
        }}
        .dir-path {{
            font-weight: 600;
            color: #333;
            font-size: 14px;
            flex: 1;
        }}
        .dir-count {{
            background: #e53e3e;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            margin-left: 10px;
        }}
        .toggle-icon {{
            color: #666;
            font-size: 12px;
            margin-right: 10px;
        }}
        .file-list {{
            display: none;
            background: white;
        }}
        .file-list.show {{
            display: block;
        }}
        .file-item {{
            padding: 15px 20px;
            border-top: 1px solid #f0f0f0;
        }}
        .file-name {{
            font-family: 'Consolas', monospace;
            color: #333;
            font-size: 13px;
            margin-bottom: 6px;
            word-break: break-all;
        }}
        .file-error {{
            color: #e53e3e;
            font-size: 12px;
            background: #fff5f5;
            padding: 6px 10px;
            border-radius: 4px;
            margin-top: 6px;
        }}
        .file-size {{
            color: #666;
            font-size: 11px;
            margin-top: 4px;
        }}
        .no-results {{
            text-align: center;
            padding: 60px 20px;
            color: #999;
        }}
        .expand-all {{
            margin-bottom: 20px;
            display: flex;
            gap: 10px;
        }}
        .btn {{
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }}
        .btn-primary {{
            background: #667eea;
            color: white;
        }}
        .btn-primary:hover {{
            background: #5568d3;
        }}
        .btn-secondary {{
            background: #e0e0e0;
            color: #333;
        }}
        .btn-secondary:hover {{
            background: #d0d0d0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>视频完整性检查报告</h1>
            <p>检查目录: {root_dir}</p>
            <p>生成时间: {timestamp} | 耗时: {elapsed_time}</p>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-label">总文件数</div>
                <div class="stat-value">{total_files:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">损坏文件</div>
                <div class="stat-value error">{corrupted_count:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">正常文件</div>
                <div class="stat-value success">{normal_count:,}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">损坏率</div>
                <div class="stat-value error">{corruption_rate:.2f}%</div>
            </div>
        </div>

        <div class="search-box">
            <input type="text" id="searchInput" placeholder="搜索文件路径或错误信息...">
        </div>

        <div class="content">
            <div class="expand-all">
                <button class="btn btn-primary" onclick="expandAll()">展开全部</button>
                <button class="btn btn-secondary" onclick="collapseAll()">折叠全部</button>
            </div>

            <div id="directories">
                {directories_html}
            </div>

            <div id="noResults" class="no-results" style="display: none;">
                <p>没有找到匹配的结果</p>
            </div>
        </div>
    </div>

    <script>
        // 搜索功能
        document.getElementById('searchInput').addEventListener('input', function(e) {{
            const searchTerm = e.target.value.toLowerCase();
            const directories = document.querySelectorAll('.directory');
            let visibleCount = 0;

            directories.forEach(dir => {{
                const dirPath = dir.querySelector('.dir-path').textContent.toLowerCase();
                const fileItems = dir.querySelectorAll('.file-item');
                let hasVisibleFiles = false;

                fileItems.forEach(item => {{
                    const fileName = item.querySelector('.file-name').textContent.toLowerCase();
                    const fileError = item.querySelector('.file-error').textContent.toLowerCase();

                    if (fileName.includes(searchTerm) || fileError.includes(searchTerm) || dirPath.includes(searchTerm)) {{
                        item.style.display = 'block';
                        hasVisibleFiles = true;
                    }} else {{
                        item.style.display = 'none';
                    }}
                }});

                if (hasVisibleFiles || searchTerm === '') {{
                    dir.style.display = 'block';
                    if (searchTerm !== '') {{
                        dir.querySelector('.file-list').classList.add('show');
                    }}
                    visibleCount++;
                }} else {{
                    dir.style.display = 'none';
                }}
            }});

            document.getElementById('noResults').style.display = visibleCount === 0 ? 'block' : 'none';
        }});

        // 折叠/展开功能
        function toggleDirectory(element) {{
            const fileList = element.nextElementSibling;
            const icon = element.querySelector('.toggle-icon');
            fileList.classList.toggle('show');
            icon.textContent = fileList.classList.contains('show') ? '▼' : '▶';
        }}

        function expandAll() {{
            document.querySelectorAll('.file-list').forEach(list => {{
                list.classList.add('show');
            }});
            document.querySelectorAll('.toggle-icon').forEach(icon => {{
                icon.textContent = '▼';
            }});
        }}

        function collapseAll() {{
            document.querySelectorAll('.file-list').forEach(list => {{
                list.classList.remove('show');
            }});
            document.querySelectorAll('.toggle-icon').forEach(icon => {{
                icon.textContent = '▶';
            }});
        }}
    </script>
</body>
</html>
"""

    # 生成目录HTML
    directories_html = []
    for dir_path in sorted(corrupted_by_dir.keys()):
        files = corrupted_by_dir[dir_path]
        files_html = []

        for file_name, error_msg, file_size in files:
            size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.1f} MB"
            files_html.append(f"""
                <div class="file-item">
                    <div class="file-name">{html.escape(file_name)}</div>
                    <div class="file-error">{html.escape(error_msg)}</div>
                    <div class="file-size">文件大小: {size_str}</div>
                </div>
            """)

        dir_html = f"""
            <div class="directory">
                <div class="dir-header" onclick="toggleDirectory(this)">
                    <div style="display: flex; align-items: center; flex: 1;">
                        <span class="toggle-icon">▶</span>
                        <span class="dir-path">{html.escape(dir_path)}</span>
                    </div>
                    <span class="dir-count">{len(files)} 个损坏文件</span>
                </div>
                <div class="file-list">
                    {''.join(files_html)}
                </div>
            </div>
        """
        directories_html.append(dir_html)

    normal_count = total_files - corrupted_count
    corruption_rate = (corrupted_count / total_files * 100) if total_files > 0 else 0

    return html_template.format(
        root_dir=html.escape(root_dir),
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        elapsed_time=elapsed_time,
        total_files=total_files,
        corrupted_count=corrupted_count,
        normal_count=normal_count,
        corruption_rate=corruption_rate,
        directories_html=''.join(directories_html) if directories_html else '<div class="no-results"><p>没有发现损坏的视频文件</p></div>'
    )

def format_time(seconds):
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}小时{minutes}分{secs}秒"
    elif minutes > 0:
        return f"{minutes}分{secs}秒"
    else:
        return f"{secs}秒"

def scan_and_check(root_dir, output_file="corrupted_videos_report.html", num_processes=None):
    """扫描并检查所有视频文件"""
    start_time = datetime.now()
    root_path = Path(root_dir)

    # 支持的视频格式
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

    print(f"开始扫描目录: {root_dir}", flush=True)
    print(f"时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("-" * 80, flush=True)

    # 收集所有视频文件
    print("正在扫描视频文件...", flush=True)
    video_files = []
    for ext in video_extensions:
        video_files.extend(root_path.rglob(f"*{ext}"))

    total = len(video_files)
    print(f"找到 {total} 个视频文件", flush=True)

    # 设置进程数
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)
    print(f"使用 {num_processes} 个进程并行检查\n", flush=True)

    # 多进程检查
    manager = Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()

    args_list = [(video_path, counter, lock) for video_path in video_files]

    corrupted = []

    print("开始检查...", flush=True)
    with Pool(processes=num_processes) as pool:
        results = pool.map(check_video_wrapper, args_list)

        for video_path, is_valid, message, file_size in results:
            if not is_valid:
                corrupted.append((video_path, message, file_size))

    print(f"\n检查完成，开始生成报告...", flush=True)

    # 按目录分组
    corrupted_by_dir = defaultdict(list)
    for video_path, error_msg, file_size in corrupted:
        dir_path = str(Path(video_path).parent)
        file_name = Path(video_path).name
        corrupted_by_dir[dir_path].append((file_name, error_msg, file_size))

    # 生成HTML报告
    elapsed = (datetime.now() - start_time).total_seconds()
    elapsed_str = format_time(elapsed)

    html_content = generate_html_report(
        corrupted_by_dir,
        total,
        len(corrupted),
        root_dir,
        elapsed_str
    )

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)

    # 总结
    print("\n" + "=" * 80, flush=True)
    print(f"检查完成！", flush=True)
    print(f"总计检查: {total} 个文件", flush=True)
    print(f"正常文件: {total - len(corrupted)} 个", flush=True)
    print(f"损坏文件: {len(corrupted)} 个 ({len(corrupted)/total*100:.2f}%)", flush=True)
    print(f"耗时: {elapsed_str}", flush=True)
    print(f"\nHTML 报告已保存到: {output_file}", flush=True)

    if corrupted:
        print(f"\n发现 {len(corrupted_by_dir)} 个目录包含损坏文件", flush=True)
        print("损坏文件最多的前 5 个目录:", flush=True)
        sorted_dirs = sorted(corrupted_by_dir.items(), key=lambda x: len(x[1]), reverse=True)
        for dir_path, files in sorted_dirs[:5]:
            print(f"  {len(files)} 个 - {dir_path}", flush=True)

if __name__ == "__main__":
    # 检查目录
    target_dir = r"G:\aws_backup\2026-09-14"
    output_report = "corrupted_videos_report.html"

    scan_and_check(target_dir, output_report)
