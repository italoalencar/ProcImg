import re
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from typing import Dict, List

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageTk
from ttkthemes import ThemedStyle

import image as img
import ui as uiclasses


class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        # self.pack()


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)

        self.label_image = tk.Label(self.root)
        self.label_image.pack()

        self.image_data = {"image": None, "file_path": None, "mode": None}
        self.image_temp = None
        self.histogram_window = None
        self.histogram = None
        self.figure_canvas_hist = FigureCanvasTkAgg(
            plt.gcf(), master=self.histogram_window
        )
        self.figure_canvas_hist.get_tk_widget().pack(
            fill="both", expand=True, side="left"
        )
        self.Dist = None

        self.C = None
        self.M = None
        self.Y = None

        self.R = None
        self.G = None
        self.B = None

        self.H = None
        self.S = None
        self.V = None

        self.image_cmy = None
        self.image_rgb = None
        self.image_hsv = None

        self.pixel_info_label = PixelInfoLabel(self.root)
        self.pixel_info_label.pack(side="bottom", fill="x")
        self.pixel_info_label.config(text=f"")
        self.image_history: List[Dict[Image.Image, str]] = []

        self.can_undo = False
        self.current_color = "I"

        self.create_dropdown_tabs()

        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack()

        self.canvas_image_id = None

    def getImageMode(self):
        return self.image_data["mode"]

    ####################################################################
    ##Manipulação da imagem
    ####################################################################

    ## Transformações
    #######
    def toNegativeView(self):
        if self.imageNotExistsAndThrowError():
            return
        image_app = self.image_data.get("image")
        negative_image = img.negative(image_app)
        self.setImage(negative_image)

    def toTLogView(self):
        if self.imageNotExistsAndThrowError():
            return
        fields_info = [("Log", "Digite a constante: ")]
        values = self.inputValues(fields_info=fields_info)
        image_app = self.image_data.get("image")
        if self.image_data.get("mode") == "grey":
            log_image = img.tLog(values[0], image_app)

        else:
            log_image = img.tLog(values[0], image_app)

        self.setImage(log_image)

    def toHistogramEqualizationView(self):
        if self.imageNotExistsAndThrowError():
            return

        image_app = self.image_data.get("image")
        if self.image_data.get("mode") == "grey":
            img_array = img.hist_equalizado(image_app)
        else:
            img_array = img.hist_equalizado_rgb(image_app)
        self.setImage(img_array)

    def toGammaCorrectionView(self):
        if self.imageNotExistsAndThrowError():
            return
        dialog = DoubleFloatDialog(
            self.root, "GamaCorrection", "Digite a constante:", "Digite o expoente:"
        )

        if dialog.result is not None:
            image_app = self.image_data["image"]
            constante, expoente = dialog.result
            log_image = img.gammaCorrection(constante, expoente, image_app)
            self.setImage(log_image)

    def toLimiarView(self):
        if self.imageNotExistsAndThrowError():
            return

        if self.image_data.get("mode") != "grey":
            self.toGrayMeanPonView()

        image_app = self.image_data["image"]
        image_processed = img.limiar(image_app)

        self.setImage(image_processed)

    def toScaleLinearView(self):
        if self.imageNotExistsAndThrowError():
            return

        dialog = DoubleFloatDialog(self.root, "Escala linear", "escala x:", "escala y:")

        if dialog.result is not None:
            image_app = self.image_data["image"]
            escala_x, escala_y = dialog.result
            negative_image = img.escala_linear(
                imagem=image_app, escala_x=escala_x, escala_y=escala_y
            )
        self.setImage(negative_image)

    def toScaleNoneView(self):
        if self.imageNotExistsAndThrowError():
            return

        dialog = DoubleFloatDialog(
            self.root, "Escala vizinho", "escala x:", "escala y:"
        )

        if dialog.result is not None:
            image_app = self.image_data["image"]
            escala_x, escala_y = dialog.result
            negative_image = img.escala_vizinho_proximo(
                imagem=image_app, escala_x=escala_x, escala_y=escala_y
            )

        self.setImage(negative_image)

    def toRotateNoneView(self):
        if self.imageNotExistsAndThrowError():
            return
        angulo = simpledialog.askfloat(
            title="Rotação vizinho:", prompt="ângulo em graus:"
        )
        image_app = self.image_data.get("image")
        negative_image = img.rotacao_vizinho_proximo(image_app, angulo=angulo)
        self.setImage(negative_image)

    def toRotateLinearView(self):
        if self.imageNotExistsAndThrowError():
            return
        angulo = simpledialog.askfloat(
            title="Rotação linear:", prompt="ângulo em graus:"
        )
        image_app = self.image_data.get("image")
        negative_image = img.rotacao_linear(image_app, angulo=angulo)
        self.setImage(negative_image)

    def toFourierView(self):
        if self.imageNotExistsAndThrowError():
            return

        image_app: Image.Image = self.image_data.get("image")
        if self.image_data.get("mode") == "grey":
            self.image_processed = img.dft(np.array(image_app.resize((32, 32))))
        else:
            self.image_processed = img.dft_rgb(np.array(image_app.resize((32, 32))))
        self.setImageFourier(
            np.array(image_app.resize((32, 32))), abs(self.image_processed)
        )
        # self.setImage(abs(image_processed))

    def toInvFourierView(self):
        if self.imageNotExistsAndThrowError():
            return
        if self.image_processed is None:
            self.showError("Imagem não transformada")
        image_app: Image.Image = self.image_data.get("image")
        # image_app = image_app.resize((32, 32))
        if self.image_data.get("mode") == "grey":
            image_processed = img.idft(self.image_processed)
        else:
            image_processed = img.idft_rgb(self.image_processed)
        self.setImageFourier(abs(self.image_processed), abs(image_processed))

    ## Modo
    #######
    def toGrayMeanAritView(self):
        if self.imageNotExistsAndThrowError():
            return
        image_app = self.image_data.get("image")
        if self.image_data.get("mode") == "rgb":
            log_image = img.imagem_cinza_media(image_app)
            self.image_data["mode"] = "grey"
            self.setImage(log_image)

        else:
            messagebox.showerror("Erro", "Imagem já está em escala de cinza")

    def toGrayMeanPonView(self):
        image_app = self.image_data.get("image")
        if self.imageNotExistsAndThrowError():
            return
        if self.image_data.get("mode") == "rgb":
            image = img.imagem_cinza_ponderada(image_app)
            self.image_data["mode"] = "grey"
            self.setImage(image)

        else:
            messagebox.showerror("Erro", "Imagem já está em escala de cinza")

    ## Filtros
    #######
    def toGrandienteView(self):
        if self.imageNotExistsAndThrowError():
            return
        image_app = self.image_data.get("image")

        if self.getImageMode() == "grey":
            image_processed = img.gradiente(image_app)
        else:
            image_processed = img.gradiente_rgb(image_app)
        self.setImage(image_processed)

    def toHighBoostView(self):
        if self.imageNotExistsAndThrowError():
            return
        image_app = self.image_data.get("image")
        fields_info = [("HighBoost", "Digite o valor da constante K: ")]
        values = self.inputValues(fields_info=fields_info)

        if self.getImageMode() == "grey":
            image_processed = img.agucamento_highboost(int(values[0]), image_app)
        else:
            image_processed = img.agucamento_highboost_rgb(int(values[0]), image_app)
        self.setImage(image_processed)

    def toLaplacianoView(self):
        if self.imageNotExistsAndThrowError():
            return
        image_app = self.image_data.get("image")

        if self.getImageMode() == "grey":
            image_processed = img.agucamento_laplace(image_app)
        else:
            image_processed = img.agucamento_laplace_rgb(image_app)
        self.setImage(image_processed)

    def toSepiaView(self):
        if self.imageNotExistsAndThrowError():
            return
        image_app = self.image_data.get("image")
        if self.getImageMode() == "grey":
            self.showError("Imagem em modo de escala de cinza")
        else:
            image_processed = img.imagem_sepia(image_app)
        self.setImage(image_processed)

    def toMeanSimpleView(self):
        if self.imageNotExistsAndThrowError():
            return
        image_app = self.image_data.get("image")

        fields_info = [("Filtro media simples", "Digite a dimensão do kernel: ")]
        values = self.inputValues(fields_info=fields_info)

        if self.getImageMode() == "grey":
            image_processed = img.suavizacao_media(int(values[0]), image_app)
        else:
            image_processed = img.suavizacao_media_rgb(int(values[0]), image_app)

        self.setImage(image_processed)

    def toMeanPonView(self):
        if self.imageNotExistsAndThrowError():
            return
        image_app = self.image_data.get("image")

        fields_info = [("Filtro media ponderada", "Digite a dimensão do kernel: ")]
        values = self.inputValues(fields_info=fields_info)
        if self.getImageMode() == "grey":
            image_processed = img.suavizacao_ponderada(int(values[0]), image_app)
        else:
            image_processed = img.suavizacao_ponderada_rgb(int(values[0]), image_app)
        self.setImage(image_processed)

    def toMedianView(self):
        if self.imageNotExistsAndThrowError():
            return
        image_app = self.image_data.get("image")

        fields_info = [("Filtro mediana", "Digite a dimensão do kernel: ")]
        values = self.inputValues(fields_info=fields_info)

        if self.getImageMode() == "grey":
            image_processed = img.suavizacao_mediana(int(values[0]), image_app)
        else:
            image_processed = img.suavizacao_mediana_rgb(int(values[0]), image_app)
        self.setImage(image_processed)

    def getandSetKernel(self, values):
        image_app = self.image_data.get("image")
        self.kernel = values

        if self.getImageMode() == "grey":
            image_processed = img.convolucao(image_app, self.kernel)
        else:
            image_processed = img.convolucao_rgb(image_app, self.kernel)
        self.setImage(image_processed)

    def toKernelFreeView(self):
        if self.imageNotExistsAndThrowError():
            return

        toKernelWindowForm = tk.Toplevel(self.root)
        toKernelWindowForm.title("Kernel free")

        toKernelWindowForm.attributes("-alpha", 0.0)
        toKernelWindowForm.minsize(500, 300)
        uiclasses.center_window(self, toKernelWindowForm)

        toKernelWindowForm.attributes("-alpha", 1.0)

        uiclasses.MatrixInputApp.create(
            toKernelWindowForm, command=self.getandSetKernel
        )

    ##Esteganografia
    def toEncryptView(self):
        if self.imageNotExistsAndThrowError():
            return
        image_app = self.image_data.get("image")
        if self.getImageMode() == "grey":
            value = simpledialog.askstring("Esteganografia", "Esteganografia")
            img_processed = img.esteganografia(image=image_app, mensagem=value)
            self.showMessage(title="Encriptado", message="A esteganografia funcionou")
            self.setImage(img_processed)
        else:
            self.showError("Imagem nao esta no modo de escala de cinza")

    def toDecryptView(self):
        if self.imageNotExistsAndThrowError():
            return
        image_app = self.image_data.get("image")
        res = img.decodificar(image_app)
        self.showMessage(title="Decriptação: ", message=res)

    ##Colors

    def on_scale_cmy_adjustment(self, values):
        image_copy = self.image_temp

        self.C = values[0]

        self.M = values[1]

        self.Y = values[2]

        # # Converta a imagem para o espaço de cores CMY
        # imagem_cmy = image_copy.convert("CMY")

        # # Converta a imagem CMY de volta para o espaço de cores RGB
        # imagem_rgb = image_copy.convert("RGB")
        imagem = np.array(image_copy)
        imagem = imagem / 255.0
        imagem[:, :, 0] = imagem[:, :, 0] - (self.C / 255)
        imagem[:, :, 1] = imagem[:, :, 1] - (self.M / 255)
        imagem[:, :, 2] = imagem[:, :, 2] - (self.Y / 255)
        imagem[imagem > 1] = 1
        imagem[imagem < 0] = 0
        imagem = (imagem * 255).astype(np.uint8)
        self.image_cmy = imagem
        # self.setImage(imagem)

    def cancelCMY(self):
        self.window_color_cmy_ad.destroy()
        self.setImage(np.array(self.image_temp))

    def onCancelCmy(self, value):
        if value:
            self.window_color_cmy_ad.destroy()
            self.setImage(np.array(self.image_temp))

    def submitCMY(self, value):
        self.window_color_cmy_ad.destroy()
        if self.image_cmy is not None:
            self.setImage(self.image_cmy)
        return

    def toColorCMYAdjustmentView(self):
        if self.imageNotExistsAndThrowError():
            return

        image_copy = self.image_data.get("image")

        self.image_temp = image_copy

        self.window_color_cmy_ad = tk.Toplevel(self.root)
        self.colorCMYAdjustment = uiclasses.ColorCMYAdjusment.create(
            parent=self.window_color_cmy_ad,
            command=self.on_scale_cmy_adjustment,
            submit=self.submitCMY,
            on_cancel=self.onCancelCmy,
        )

    def on_scale_adjustment(self, values):
        image_copy = self.image_temp
        # img_hsv = img.imagem_rgb_to_hsv(image_copy)

        R = values[0]
        G = values[1]
        B = values[2]

        imagem = np.array(image_copy)
        imagem = imagem / 255.0
        imagem[:, :, 0] = imagem[:, :, 0] + (R / 255)
        imagem[:, :, 1] = imagem[:, :, 1] + (G / 255)
        imagem[:, :, 2] = imagem[:, :, 2] + (B / 255)
        imagem[imagem > 1] = 1
        imagem = (imagem * 255).astype(np.uint8)
        self.image_rgb = imagem
        # self.setImage(imagem)

    def cancelRGB(self):
        self.window_color_ad.destroy()
        self.setImage(np.array(self.image_temp))

    def onCancelRgb(self, value):
        if value:
            self.window_color_ad.destroy()
            self.setImage(np.array(self.image_temp))

    def submitRGB(self, value):
        self.window_color_ad.destroy()
        if self.image_rgb is not None:
            self.setImage(self.image_rgb)
        return

    def toColorAdjustmentView(self):
        if self.imageNotExistsAndThrowError():
            return

        image_copy = self.image_data.get("image")

        self.image_temp = image_copy

        self.window_color_ad = tk.Toplevel(self.root)
        self.colorAdjustment = uiclasses.ColorAdjusment.create(
            parent=self.window_color_ad,
            command=self.on_scale_adjustment,
            submit=self.submitRGB,
            on_cancel=self.onCancelRgb,
        )

    def on_hsv_adjustment(self, values):
        self.H = values[0]
        self.S = values[1]
        self.V = values[2]

        # self.setImage(img_rgb_temp)

    def cancelHsv(self):
        self.window_hsv_ad.destroy()
        self.setImage(np.array(self.image_temp))

    def onCancelHsv(self, value):
        if value:
            self.window_hsv_ad.destroy()
            self.setImage(np.array(self.image_temp))

    def submitHSV(self, value):
        image_copy = self.image_temp
        image = np.array(image_copy)
        image = image / 255.0
        img_hsv = img.imagem_rgb_to_hsv(image)

        img_hsv[:, :, 0] = img_hsv[:, :, 0] + self.H / 255
        img_hsv[:, :, 1] = img_hsv[:, :, 1] + self.S / 255
        img_hsv[:, :, 2] = img_hsv[:, :, 2] + self.V / 255

        img_rgb_temp = img.imagem_hsv_to_rgb(img_hsv)
        img_rgb_temp = (img_rgb_temp * 255).astype(np.uint8)
        self.image_hsv = img_rgb_temp

        self.setImage(img_rgb_temp)
        self.window_hsv_ad.destroy()
        return

    def toHSVAdjustmentView(self):
        if self.imageNotExistsAndThrowError():
            return

        image_copy = self.image_data.get("image")

        self.image_temp = image_copy

        self.window_hsv_ad = tk.Toplevel(self.root)
        self.hsvAdjustment = uiclasses.HSVAdjusment.create(
            parent=self.window_hsv_ad,
            command=self.on_hsv_adjustment,
            submit=self.submitHSV,
            on_cancel=self.onCancelHsv,
        )

        self.window_hsv_ad.protocol("WM_DELETE_WINDOW", func=self.cancelHsv)

    def on_chroma(self, values):
        self.Dist = values[0]

    def cancelChroma(self):
        self.window_color_ad.destroy()
        self.setImage(np.array(self.image_temp))

    def onCancelChroma(self, value):
        if value:
            self.window_chroma_ad.destroy()
            self.setImage(np.array(self.image_temp))

    def submitChroma(self, value):
        self.window_chroma_ad.destroy()
        if self.imagechroma is None:
            self.showError("Imagem de fundo não carregada/valida")
        if self.Dist is not None:
            image_app = self.image_data.get("image")
            image_processed = img.chromakey(self.imagechroma, image_app, self.Dist)
            self.setImage(image_processed)
            # self.mostrar_imagem(np.array(image_processed))
        return

    def toChromaKeyView(self):
        if self.imageNotExistsAndThrowError():
            return

        file_path = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.bpm;*.jpg;*.jpeg;*webp;")]
        )
        if file_path:
            image = img.open_image(file_path)
            self.imagechroma = image
        else:
            return

        image_copy = self.image_data.get("image")

        self.image_temp = image_copy

        self.window_chroma_ad = tk.Toplevel(self.root)
        self.colorAdjustment = uiclasses.ChromaAdjusment.create(
            parent=self.window_chroma_ad,
            command=self.on_chroma,
            submit=self.submitChroma,
            on_cancel=self.onCancelChroma,
        )

    def mostrar_imagem(self, original):
        fig, ax = plt.subplots(ncols=2, figsize=(15, 5))
        ax[0].imshow(original, cmap="gray")
        ax[0].set_title("Original")
        ax[0].axis("off")
        plt.show()

    ####################################################################
    ##Actions
    ####################################################################
    def openImage(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.bpm;*.jpg;*.jpeg;")]
        )
        if file_path:
            image = img.open_image(file_path)
            image_tk = ImageTk.PhotoImage(image)
            self.canvas_image_id = self.canvas.create_image(
                0, 0, anchor=tk.NW, image=image_tk
            )
            self.canvas.image = image_tk

            self.image_data["file_path"] = file_path

            self.image_data["image"] = image
            self.image_data["mode"] = "rgb"
            self.image_history.append({"image": image, "mode": "rgb"})
            self.pixel_info_label.configure(
                text=f"Dimensões: {image.width} x {image.height}"
            )
            self.centralizar_imagem()

    def saveImage(self):
        current_image = self.image_data.get("image")
        if current_image:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".jpg",
                filetypes=[("Image Files", "*.bpm;*.jpg;*.jpeg;")],
            )
            if save_path:
                current_image.save(save_path)
                messagebox.showinfo("Salvo", f"Imagem salva como {save_path}")
        else:
            self.showError("Não foi possivel salvar a imagem.")

    def showHistogram(self):
        if self.imageNotExistsAndThrowError():
            return
        image_app: Image.Image = self.image_data.get("image")

        # Mostrar o histograma com base na cor selecionada
        self.plot_histogram(image_app, self.current_color)

    def imageNotExistsAndThrowError(self):
        image_app = self.image_data.get("image")
        if image_app is None:
            messagebox.showerror("Erro", "Nenhuma imagem carregada")
            return True
        return False

    def showErrorGray(self, function):
        if self.image_data["mode"] == "grey":
            function()
        else:
            messagebox.showerror("Erro", "Imagem não está em escala de cinza")

    def showError(self, message: str):
        messagebox.showerror("Erro", message)

    def showMessage(self, title: str, message: str):
        messagebox.showinfo(title, message)

    def on_histogram_window_close(self):
        if self.histogram_window is not None:
            self.histogram_window.destroy()
            self.histogram_window = None

    def on_radio_button_select(self):
        self.current_color = self.radio.get_selected_value()
        self.update_histogram()
        self.plot_histogram(self.image_data["image"], self.current_color)

    def update_histogram(self):
        # if self.histogram_window is not None:
        if self.imageNotExistsAndThrowError():
            return

        image_app: Image.Image = self.image_data.get("image")

        if self.image_data["mode"] != "grey":
            histograms = img.histograma_RGBI(image_app)
            match self.current_color:
                case "R":
                    image_dic = histograms[0]
                case "G":
                    image_dic = histograms[1]
                case "B":
                    image_dic = histograms[2]
                case "I":
                    image_dic = histograms[3]
                case _:
                    image_dic = histograms[3]
        else:
            histograms = [img.histograma(image_app)]
            image_dic = histograms[0]

        self.histogram = image_dic

        if self.histogram_window is not None:
            self.plot_histogram(self.image_data["image"], self.current_color)

    def plot_histogram(self, image: Image.Image, color):
        if self.histogram is None:
            self.update_histogram()

        plt.clf()

        match self.current_color:
            case "R":
                color_bar = "red"
            case "G":
                color_bar = "green"
            case "B":
                color_bar = "blue"
            case "I":
                color_bar = "grey"
            case _:
                color_bar = "grey"

        intensidade = list(self.histogram.keys())
        ocorrencia = list(self.histogram.values())

        plt.bar(intensidade, ocorrencia, color=color_bar)
        plt.xlabel("Intensidade")
        plt.ylabel("Ocorrencia")
        plt.title("Histograma")

        if self.histogram_window is None:
            self.histogram_window = tk.Toplevel(self.root)
            self.histogram_window.title("Histograma")

            options = ["R", "G", "B", "I"]
            self.radio = uiclasses.RadioButtons.create(
                self.histogram_window, options, options[3], self.on_radio_button_select
            )

        if self.figure_canvas_hist is not None:
            # Limpe a figura anterior do canvas
            self.figure_canvas_hist.get_tk_widget().pack_forget()

        # Crie um novo FigureCanvasTkAgg e coloque-o no mesmo local
        self.figure_canvas_hist = FigureCanvasTkAgg(
            plt.gcf(), master=self.histogram_window
        )
        self.figure_canvas_hist.get_tk_widget().pack(
            fill="both", expand=True, side="left"
        )

        # if(self.histogram_window is None):

        #                 # Antes de criar um novo gráfico, destrua o gráfico anterior
        #     if self.figure_canvas_hist is not None:
        #         self.figure_canvas_hist.get_tk_widget().get_tk_widget().destroy()

        #     self.histogram_window = tk.Toplevel(self.root)
        #     self.histogram_window.title("Histograma")

        #     self.figure_canvas_hist = FigureCanvasTkAgg(plt.gcf(), master=self.histogram_window)
        #     self.figure_canvas_hist.draw()

        #

        self.histogram_window.protocol(
            "WM_DELETE_WINDOW", self.on_histogram_window_close
        )

        #     self.figure_canvas_hist.get_tk_widget().pack(fill="both", expand=True, side="left")
        # else:
        #     self.figure_canvas_hist = FigureCanvasTkAgg(plt.gcf(), master=self.histogram_window)
        #     self.figure_canvas_hist.draw()
        #     self.figure_canvas_hist.get_tk_widget().pack(fill="both", expand=True, side="left")

    def set_mode_rgb(self):
        print("rgb")

    ####################################################################
    ## UI
    ####################################################################
    def centralizar_imagem(self, event=None):
        if self.image_data.get("image") is None:
            return
        # Recupere as dimensões da janela
        janela_width = self.root.winfo_width()
        janela_height = self.root.winfo_height()

        # Recupere as dimensões da imagem
        image_width, image_height = self.image_data.get("image").size

        # Calcule as coordenadas X e Y para centralizar a imagem
        x = (janela_width - image_width) // 2
        y = (janela_height - image_height) // 2

        # Reposicione o widget Label que exibe a imagem
        self.canvas.place(x=x, y=y)

    def create_dropdown_tabs(self):
        tab_texts = [
            "Arquivo",
            "Transformações",
            "Modo",
            "Editar",
            "Exibir",
            "Ajuda",
            "Filtros",
            "Esteganografia",
            "Cores",
        ]

        tab_menu_file = tk.Menu(self.menubar, tearoff=0)

        self.menubar.add_cascade(label=tab_texts[0], menu=tab_menu_file)

        self.add_option_to_tab(tab_menu_file, "Abrir imagem", self.openImage)
        self.add_option_to_tab(tab_menu_file, "Salvar imagem", self.saveImage)

        tab_menu_transform = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=tab_texts[1], menu=tab_menu_transform)

        self.add_option_to_tab(tab_menu_transform, "Negative", self.toNegativeView)

        self.add_option_to_tab(tab_menu_transform, "Log", self.toTLogView)

        self.add_option_to_tab(tab_menu_transform, "Gamma", self.toGammaCorrectionView)

        self.add_option_to_tab(tab_menu_transform, "Limiarização", self.toLimiarView)

        self.add_option_to_tab(
            tab_menu_transform, "Rotação Linear", self.toRotateLinearView
        )

        self.add_option_to_tab(
            tab_menu_transform, "Rotação Vizinho", self.toRotateNoneView
        )

        self.add_option_to_tab(
            tab_menu_transform, "Escala Linear", self.toScaleLinearView
        )

        self.add_option_to_tab(
            tab_menu_transform, "Escala Vizinho", self.toScaleNoneView
        )

        self.add_option_to_tab(
            tab_menu_transform, "Transformada de Fourier", self.toFourierView
        )

        self.add_option_to_tab(
            tab_menu_transform, "Inversa de Fourier", self.toInvFourierView
        )

        self.add_option_to_tab(
            tab_menu_transform,
            "Histograma equalizado",
            self.toHistogramEqualizationView,
        )

        tab_menu_filters = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=tab_texts[6], menu=tab_menu_filters)

        self.add_option_to_tab(
            tab_menu_filters, "Aguçamento Laplaciano", self.toLaplacianoView
        )
        self.add_option_to_tab(tab_menu_filters, "High-boost", self.toHighBoostView)
        self.add_option_to_tab(tab_menu_filters, "Gradiente", self.toGrandienteView)
        self.add_option_to_tab(tab_menu_filters, "Sepia", self.toSepiaView)
        self.add_option_to_tab(tab_menu_filters, "Media ponderada", self.toMeanPonView)
        self.add_option_to_tab(tab_menu_filters, "Media simples", self.toMeanSimpleView)
        self.add_option_to_tab(tab_menu_filters, "Mediana", self.toMedianView)
        self.add_option_to_tab(tab_menu_filters, "Convolução", self.toKernelFreeView)

        tab_menu_colors = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=tab_texts[8], menu=tab_menu_colors)

        self.add_option_to_tab(tab_menu_colors, "Chroma key", self.toChromaKeyView)
        self.add_option_to_tab(
            tab_menu_colors, "Color R/G/B adjusment", self.toColorAdjustmentView
        )
        self.add_option_to_tab(
            tab_menu_colors, "Color C/M/Y adjusment", self.toColorCMYAdjustmentView
        )
        self.add_option_to_tab(tab_menu_colors, "HSV", self.toHSVAdjustmentView)

        tab_menu_estegono = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label=tab_texts[7], menu=tab_menu_estegono)

        self.add_option_to_tab(tab_menu_estegono, "Encriptar", self.toEncryptView)
        self.add_option_to_tab(tab_menu_estegono, "Descriptar", self.toDecryptView)

        tab_menu_mode = tk.Menu(self.menubar, tearoff=0)

        self.menubar.add_cascade(label=tab_texts[2], menu=tab_menu_mode)

        sub_menu_mode = tk.Menu(tab_menu_mode, tearoff=0)
        tab_menu_mode.add_cascade(label="Tons de cinza", menu=sub_menu_mode)

        self.add_option_to_tab(sub_menu_mode, "Media ponderada", self.toGrayMeanPonView)
        self.add_option_to_tab(
            sub_menu_mode, "Media aritmetica", self.toGrayMeanAritView
        )

        # self.add_option_to_tab(tab_menu_mode, "RGB", self.set_mode_rgb)

        tab_menu_edit = tk.Menu(self.menubar, tearoff=0)

        self.menubar.add_cascade(label=tab_texts[3], menu=tab_menu_edit)
        self.add_option_to_tab(tab_menu_edit, "Desfazer", self.undo)

        tab_menu_view = tk.Menu(self.menubar, tearoff=0)

        self.menubar.add_cascade(label=tab_texts[4], menu=tab_menu_view)
        self.add_option_to_tab(tab_menu_view, "Histograma", self.showHistogram)

        # self.add_option_to_tab(tab_menu_file, "Abrir Gráfico", self.openGraph)

    def add_option_to_tab(self, tab_menu, text_option, command_tab):
        tab_menu.add_command(label=text_option, command=command_tab)

    def showImage(self, image=None):
        if image is not None:
            # Obtenha as dimensões da imagem
            height, width, channels = image.shape

            # Crie um objeto PhotoImage do Tkinter
            image_tk = tk.PhotoImage(width=width, height=height)

            # Preencha o objeto PhotoImage com os dados da imagem
            for y in range(height):
                for x in range(width):
                    pixel = tuple(image[y, x])
                    color = "#{:02x}{:02x}{:02x}".format(*pixel)
                    image_tk.put(color, (x, y))

            # Exibe a imagem no canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
            self.canvas.image = image_tk
        else:
            image = self.image_data.get("image")

            # Obtenha as dimensões da imagem
            height, width, channels = image.shape

            # Crie um objeto PhotoImage do Tkinter
            image_tk = tk.PhotoImage(width=width, height=height)

            # Preencha o objeto PhotoImage com os dados da imagem
            for y in range(height):
                for x in range(width):
                    r, g, b = image[y, x]

                    # Garanta que os valores estejam no intervalo [0, 255]
                    r = int(max(0, min(255, r)))
                    g = int(max(0, min(255, g)))
                    b = int(max(0, min(255, b)))

                    # Formate como representação hexadecimal
                    color = "#{:02x}{:02x}{:02x}".format(r, g, b)

                    image_tk.put(color, (x, y))

            # Exibe a imagem no canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
            self.canvas.image = image_tk

    def setImage(self, image_arr):
        image = Image.fromarray(image_arr)
        image_tk = ImageTk.PhotoImage(image)

        # Atualize o canvas com a nova imagem
        self.canvas.itemconfig(self.canvas_image_id, image=image_tk)

        # Atribua a nova imagem à variável do canvas
        self.canvas.image = image_tk
        self.image_data["image"] = image

        mode = self.image_data["mode"]

        self.pixel_info_label.configure(
            text=f"Dimensões: {image.width} x {image.height}"
        )
        self.image_history.append({"image": image, "mode": mode})
        self.can_undo = True

    def setImageFourier(self, original, modificada):
        plt.clf()
        fig, ax = plt.subplots(ncols=2, figsize=(15, 5))
        ax[0].imshow(original, cmap="gray")
        ax[0].set_title("Original")
        ax[0].axis("off")
        ax[1].imshow(modificada, cmap="gray")
        ax[1].set_title("Fourier")
        ax[1].axis("off")
        plt.show()

    def undo(self):
        if len(self.image_history) > 1:
            self.image_history.pop()

            image_history = self.image_history[-1]
            previous_image = image_history.get("image")
            previous_mode = image_history.get("mode")
            image_tk = ImageTk.PhotoImage(previous_image)

            self.canvas.itemconfig(self.canvas_image_id, image=image_tk)

            self.canvas.image = image_tk
            self.image_data["image"] = previous_image
            self.image_data["mode"] = previous_mode
            self.update_histogram()

    def inputValues(self, fields_info):
        values = []
        for text, placeholder in fields_info:
            value = simpledialog.askfloat(text, placeholder)
            if value is not None:
                values.append(float(value))
        return values

    def validar_matriz(self, entrada):
        return re.match(
            r"^\s*\[\s*(\[\s*\d+\s*(,\s*\d+\s*)*\]\s*,?)*\s*\]\s*$", entrada
        )

    # def on_mouse_move(self, event):
    #     image = self.image_data.get("image")
    #     if(image == None):
    #         return
    #     x, y = event.x, event.y
    #     img_x = x
    #     img_y = y
    #     mode = self.image_data.get("mode")
    #     try:
    #         pixel_value = image.getpixel((img_x, img_y))
    #         if mode == "grey":
    #             self.pixel_info_label.config(text=f"Coordenadas: ({img_x}, {img_y}), Intensidade de Cinza: {pixel_value}")
    #         elif mode == "rgb":
    #             r, g, b = pixel_value
    #             self.pixel_info_label.config(text=f"Coordenadas: ({img_x}, {img_y}), Cor: RGB({r}, {g}, {b})")
    #     except IndexError:
    #         self.pixel_info_label.config(text="Coordenadas fora da imagem")

    def plot_line(self):
        self.ax.clear()
        self.ax.plot(self.x, self.y, label="Reta")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.legend()
        self.canvas.draw()

    def update_line(self, slope):
        self.y = (
            float(slope) * self.x
        )  # Atualize a função da reta com base no controle deslizante
        self.plot_line()


class ViewImage(tk.Label):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()


class PixelInfoLabel(tk.Label):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)


class DoubleFloatDialog(simpledialog.Dialog):
    def __init__(self, parent, title, prompt1, prompt2):
        self.prompt1 = prompt1
        self.prompt2 = prompt2
        super().__init__(parent, title=title)

    def body(self, master):
        tk.Label(master, text=self.prompt1).grid(row=0)
        tk.Label(master, text=self.prompt2).grid(row=1)
        self.entry1 = tk.Entry(master)
        self.entry2 = tk.Entry(master)
        self.entry1.grid(row=0, column=1)
        self.entry2.grid(row=1, column=1)
        return self.entry1

    def apply(self):
        self.result = (float(self.entry1.get()), float(self.entry2.get()))


if __name__ == "__main__":
    # create the application
    myapp = App()

    myapp.master.title("Processador de imagens")
    # myapp.master.maxsize(1000, 400)
    # myapp.master.minsize(900, 600)
    myapp.master.geometry("1000x600")

    view_image = ImageApp(myapp.master)

    # view_image.root.bind("<Motion>", view_image.on_mouse_move)
    myapp.master.bind("<Configure>", view_image.centralizar_imagem)
    # myapp.master.protocol("WM_DELETE_WINDOW", self.on_histogram_window_close)
    # view_image.histogram_window.protocol("WM_DELETE_WINDOW", view_image.on_histogram_window_close)
    # start the program
    # myapp.master.mainloop()
    myapp.master.state("zoomed")
    myapp.mainloop()
