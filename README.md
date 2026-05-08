# Deploying-Tkinter-based-GUI-AI-code-exe-using-Python
Deploying Tkinter-based GUI AI code (exe) using Python


PyTorch训练(.pth)
↓
pth的验证
↓
导出ONNX(.onnx)
↓
验证ONNX正确性
输出模型的结果的形状大小，与预期匹配即可      
↓
Python GUI加载ONNX
↓
PyInstaller打包exe（把onnx和所需的数据都放在同一个文件夹里）
pyinstaller -F -w GEO_PowerPredicted_GUI_v4.py --exclude-module PyQt5 --exclude-module PyQt6
↓
Windows直接运行
