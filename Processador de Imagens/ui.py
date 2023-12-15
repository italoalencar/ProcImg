import tkinter as tk
from tkinter import ttk, messagebox
import re

class MatrixInputApp:
    def __init__(self, parent: tk.Tk, command):
        
        self.parent = parent
        self.parent.title("Matriz")
        self.command = command    
        self.tamanho_matriz = tk.StringVar() 
        self.tamanho_matriz.set("3x3")
    
        self.matriz_resultante = []
        self.matriz_formatada = []
        self.open = False
        self.input_window : tk.Toplevel = None
        self.create_ui()

    @staticmethod
    def create(parent, command):
        return MatrixInputApp(parent, command= command)
    
    def create_ui(self):
        self.center_window(self.parent)
        self.select_label = tk.Label(self.parent, text="Selecione o tamanho da matriz:")
        self.select_label.pack()

        self.tamanho_combobox = ttk.Combobox(self.parent, values=["3x3", "5x5", "7x7", "9x9"], textvariable=self.tamanho_matriz)
        self.tamanho_combobox.pack()

        self.confirm_button = tk.Button(self.parent, text="Confirmar", command=self.generate_matrix)
        self.confirm_button.pack()

    def center_window(self, window):
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()

        x = (screen_width - window.winfo_reqwidth()) / 2
        y = (screen_height - window.winfo_reqheight()) / 2

        window.geometry("+%d+%d" % (x, y))

    def convert_to_float(self, input_string):
        if re.match(r'^\d+,\d+$', input_string):
            input_string = input_string.replace(',', '.')

        try:
            result = float(input_string)
            return result
        except ValueError:
            return None

    def generate_matrix(self):
        if self.open : return
        selected_size = self.tamanho_matriz.get()
        size = int(selected_size.split('x')[0])

        self.input_window = tk.Toplevel(self.parent)
        self.input_window.title("Preencha a matriz")
        self.center_window(self.input_window)
        self.open = True
        self.input_window.protocol("WM_DELETE_WINDOW", self.closedInput )

        self.matriz_resultante : list = []
        self.matriz_formatada = []

        for i in range(size):
            row = []
            for j in range(size):
                label = tk.Label(self.input_window, text=f"Valor [{i+1}][{j+1}]:")
                label.grid(row=i, column=j, padx=5, pady=5)

                entry = tk.Entry(self.input_window)
                entry.grid(row=i, column=j, padx=5, pady=5)

                row.append(entry)
                
            self.matriz_resultante.append(row)
            
        validate_button = tk.Button(self.input_window, text="Validar", command=self.validate_inputs)
        validate_button.grid(row=size, columnspan=size, padx=5, pady=10)


    def closedInput(self):
        self.open = False
        self.input_window.destroy()

    def validate_inputs(self):
        if self.matriz_resultante is not None:
            for row in self.matriz_resultante:
                row_formated=[]
                for entry in row:
                    value = entry.get()
                    if not value:
                        messagebox.showerror("Erro", "Todos os campos devem ser preenchidos")
                        return

                    if not (self.validar_numeros(value)):
                        tk.messagebox.showerror("Erro", "Todos os campos devem ser numéricos")
                        self.matriz_formatada = []
                        return
                    row_formated.append(self.convert_to_float(entry.get()))
                    
                self.matriz_formatada.append(row_formated)
            
            self.parent.matriz_conv = self.matriz_formatada
            self.command(self.matriz_formatada)
            messagebox.showinfo("Sucesso", "Matriz gerada e validada com sucesso!")
            self.open = False
            self.parent.destroy()
        else:
            messagebox.showerror("Erro", "A matriz ainda não foi gerada.")
    
    def validar_numeros(self, string):
        regex = r"^-?\d+(\.\d+)?(,\d+)?$"
        return bool(re.match(regex, string))

def center_window(self, window):
    
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()

    x = (screen_width - window.winfo_reqwidth()) / 2
    y = (screen_height - window.winfo_reqheight()) / 2

    window.geometry("+%d+%d" % (x, y))

class RadioButtons:
    def __init__(self, master, options, value, command):
        self.master = master
        self.value = tk.StringVar(value=value)
        self.command = command
        self.radio_buttons = []

        for i, option in enumerate(options):
            radio_button = tk.Radiobutton(master, text=option, variable=self.value, value=option, command=self.command)
            
            radio_button.pack(side=tk.LEFT, padx=5)
            self.radio_buttons.append(radio_button)

    def get_selected_value(self):
        return self.value.get()

    @staticmethod
    def create(master, options, value, command):
        return RadioButtons(master, options, value, command)


class ColorAdjusment:
    def __init__(self, parent, command, submit, on_cancel):
        self.parent = parent
        self.value = None
        self.command = command
        self.submit = submit
        self.on_cancel = on_cancel
        self.scales = []

        
        
        label1 = tk.Label(parent, text= "R")
        label1.grid(row=0, column=0, padx=5)
        self.scale1 = tk.Scale(parent, from_=0, to=255, orient="horizontal", command = self.getData )
        self.scale1.grid(row=0, column=1, padx=5)

        label2 = tk.Label(parent,text= "G")
        label2.grid(row=1, column=0, padx=5)
        self.scale2 = tk.Scale(parent, from_=0, to=255, orient="horizontal", command = self.getData )
        self.scale2.grid(row=1, column=1, padx=5)

        label3 = tk.Label(parent,text= "B")
        label3.grid(row=2, column=0, padx=5)
        self.scale3 = tk.Scale(parent, from_=0, to=255, orient="horizontal", command = self.getData )
        self.scale3.grid(row=2, column=1, padx=5)


        self.scales.append(self.scale1)
        self.scales.append(self.scale2)
        self.scales.append(self.scale3)

        self.button1 = tk.Button(parent, text="Aplicar", command = self.apply)
        self.button2 = tk.Button(parent, text="Cancelar", command = self.cancel)
        self.button1.grid(row=3, column=0, padx=5)
        self.button2.grid(row=3, column=1, padx=5)

    def getData(self, value):
            self.command([self.scale1.get(), self.scale2.get(), self.scale3.get()])
        
    def apply(self):
        self.submit([self.scale1.get(), self.scale2.get(), self.scale3.get()])

    def cancel(self):
        self.on_cancel(True)

    def get_values(self):
        return self.value
    
    @staticmethod
    def create(parent, command, submit, on_cancel):
        return ColorAdjusment(parent, command, submit, on_cancel)



class ColorCMYAdjusment:
    def __init__(self, parent, command, submit, on_cancel):
        self.parent = parent
        self.value = None
        self.command = command
        self.submit = submit
        self.on_cancel = on_cancel
        self.scales = []

        label1 = tk.Label(parent, text= "C")
        label1.grid(row=0, column=0, padx=5)
        self.scale1 = tk.Scale(parent, from_=0, to=255, orient="horizontal", command = self.getData )
        self.scale1.grid(row=0, column=1, padx=5)

        label2 = tk.Label(parent,text= "M")
        label2.grid(row=1, column=0, padx=5)
        self.scale2 = tk.Scale(parent, from_=0, to=255, orient="horizontal", command = self.getData )
        self.scale2.grid(row=1, column=1, padx=5)

        label3 = tk.Label(parent,text= "Y")
        label3.grid(row=2, column=0, padx=5)
        self.scale3 = tk.Scale(parent, from_=0, to=255, orient="horizontal", command = self.getData )
        self.scale3.grid(row=2, column=1, padx=5)


        self.scales.append(self.scale1)
        self.scales.append(self.scale2)
        self.scales.append(self.scale3)

        self.button1 = tk.Button(parent, text="Aplicar", command = self.apply)
        self.button2 = tk.Button(parent, text="Cancelar", command = self.cancel)
        self.button1.grid(row=3, column=0, padx=5)
        self.button2.grid(row=3, column=1, padx=5)

    def getData(self, value):
            self.command([self.scale1.get(), self.scale2.get(), self.scale3.get()])
        
    def apply(self):
        self.submit([self.scale1.get(), self.scale2.get(), self.scale3.get()])

    def cancel(self):
        self.on_cancel(True)

    def get_values(self):
        return self.value
    
    @staticmethod
    def create(parent, command, submit, on_cancel):
        return ColorCMYAdjusment(parent, command, submit, on_cancel)




class HSVAdjusment:
    def __init__(self, parent,  command, submit, on_cancel):
        self.parent = parent
        self.value = None
        self.command = command
        self.submit = submit
        self.on_cancel = on_cancel
        self.scales = []

        label1 = tk.Label(parent, text= "H")
        label1.grid(row=0, column=0, padx=5)
        self.scale1 = tk.Scale(parent, from_=0, to=360, orient="horizontal", command = self.getData )
        self.scale1.grid(row=0, column=1, padx=5)

        label2 = tk.Label(parent,text= "S")
        label2.grid(row=1, column=0, padx=5)
        self.scale2 = tk.Scale(parent, from_=0, to=100, orient="horizontal", command = self.getData )
        self.scale2.grid(row=1, column=1, padx=5)

        label3 = tk.Label(parent,text= "V")
        label3.grid(row=2, column=0, padx=5)
        self.scale3 = tk.Scale(parent, from_=0, to=100, orient="horizontal", command = self.getData )
        self.scale3.grid(row=2, column=1, padx=5)
          
        self.scales.append(self.scale1)
        self.scales.append(self.scale2)
        self.scales.append(self.scale3)

        self.button1 = tk.Button(parent, text="Aplicar", command = self.apply)
        self.button2 = tk.Button(parent, text="Cancelar", command = self.cancel)
        self.button1.grid(row=3, column=0, padx=5)
        self.button2.grid(row=3, column=1, padx=5)
        
    def getData(self, value):
        self.command([self.scale1.get(), self.scale2.get(), self.scale3.get()])
    
    def apply(self):
        self.submit([self.scale1.get(), self.scale2.get(), self.scale3.get()])

    def cancel(self):
        self.on_cancel(True)

    def get_values(self):
        return self.value
    
    @staticmethod
    def create(parent, command, submit, on_cancel):
        return HSVAdjusment(parent, command, submit, on_cancel)




class ChromaAdjusment:
    def __init__(self, parent,  command, submit, on_cancel):
        self.parent = parent
        self.value = None
        self.command = command
        self.submit = submit
        self.on_cancel = on_cancel
        self.scales = []

       
        label4 = tk.Label(parent,text= "Distancia")
        label4.grid(row=0, column=0, padx=5)
        self.scale4 = tk.Scale(parent, from_=0, to=100, orient="horizontal", command = self.getData )
        self.scale4.grid(row=0, column=1, padx=5)
      
        self.scales.append(self.scale4)

        self.button1 = tk.Button(parent, text="Aplicar", command = self.apply)
        self.button2 = tk.Button(parent, text="Cancelar", command = self.cancel)
        self.button1.grid(row=1, column=0, padx=5)
        self.button2.grid(row=1, column=1, padx=5)
        
    def getData(self, value):
        self.command([self.scale4.get()])
    
    def apply(self):
        self.submit([self.scale4.get()])

    def cancel(self):
        self.on_cancel(True)

    def get_values(self):
        return self.value
    
    @staticmethod
    def create(parent, command, submit, on_cancel):
        return ChromaAdjusment(parent, command, submit, on_cancel)
    