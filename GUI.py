from Tkinter import Tk, Label, Button
import Tkinter as tk
import matplotlib
import numpy as np 
import matplotlib.pyplot as plt 
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg

class MyFirstGUI:
    def __init__(self, master):
        self.master = master
        master.title("Explore Image")

        self.label = Label(master, text="This is our first GUI!")
        self.label.pack()

        self.greet_button = Button(master, text="Greet", command=self.greet)
        self.greet_button.pack()

        
        f = plt.figure(figsize=(5,5))
        self.image_plot = f.add_subplot(111)
        self.image_plot.plot(np.arange(5), np.arange(5), 'r:')
        canvas = FigureCanvasTkAgg(f, self)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2TkAgg(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()

    def greet(self):
        print("Greetings!")

root = Tk()
my_gui = MyFirstGUI(root)
root.mainloop()