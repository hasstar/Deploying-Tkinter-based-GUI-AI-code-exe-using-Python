import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import onnxruntime as ort

def load_csv():
    file_path = filedialog.askopenfilename()
    if file_path:
        data = np.loadtxt(file_path, delimiter=',')
        return data
# ===== 模型加载 =====
sess = ort.InferenceSession("AWPL_checkpoints_300_900_30_Fusion_W2_model.onnx")

Y = 10   # ⚠️改成你的
atm_dim = 241  # ⚠️改成你的

# ===== 推理函数 =====
def run_model():
    try:
        x = load_csv()   # 你的功率
        a = load_csv()   # 你的大气
        x = x.reshape(1, Y, 1).astype(np.float32)
        a = a.reshape(1, atm_dim).astype(np.float32)

        # x = np.random.randn(1, Y, 1).astype(np.float32)
        # a = np.random.randn(1, atm_dim).astype(np.float32)

        out = sess.run(None, {
            "power_seq": x,
            "atm": a
        })

        result = out[0]
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, str(result))

    except Exception as e:
        messagebox.showerror("错误", str(e))

# ===== GUI =====
root = tk.Tk()
root.title("卫星信道预测系统")
root.geometry("500x400")

btn = tk.Button(root, text="运行预测", command=run_model)
btn.pack(pady=20)

output_text = tk.Text(root, height=15)
output_text.pack()

root.mainloop()