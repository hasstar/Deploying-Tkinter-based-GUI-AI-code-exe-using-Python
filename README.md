# Deploying-Tkinter-based-GUI-AI-code-exe-using-Python
Deploying Tkinter-based GUI AI code (exe) using Python

注意，本工程不使用QT包，使用TKinter部署。


PyTorch训练(.pth)

模型训练的时候生成

↓

pth的验证

A_TAP_Load_app.py 利用pth文件进行模型部署

↓

导出ONNX(.onnx)

A_TAP_ModelDeployment_step1.py

↓

验证ONNX正确性

A_TAP_ModelDeployment_step2.py

输出模型的结果的形状大小，与预期匹配即可      

↓

Python GUI加载ONNX

A_TAP_ModelDeployment_step3.py

GEO_PowerPredicted_GUI_v4.py

↓

PyInstaller打包exe（把onnx和所需的数据都放在同一个文件夹里）

pyinstaller -F -w GEO_PowerPredicted_GUI_v4.py --exclude-module PyQt5 --exclude-module PyQt6

↓

Windows直接运行
