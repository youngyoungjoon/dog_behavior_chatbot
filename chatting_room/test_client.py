from tkinter import *
import socket
import threading

HOST = '127.0.0.1'
PORT = 22
NAME = ""
flag = True

class myApp:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    def __init__(self, Master):
        self.master = Master
        Master.geometry("400x600")
        self.masterFrame = Frame(Master)
        self.masterFrame.pack(fill=X)

        self.frame1 = Frame(self.masterFrame)
        self.frame1.pack(fill=X)

        self.messageLog = Text(self.frame1)
        self.scrollbar = Scrollbar(self.frame1, command=self.messagelog.yview)
        self.messageLog.configure(width=56, height=40, state="disabled", yscrollcommand=self.scrollbar.set)
        self.messageLog.gird()
        self.messageLog.pack(side=LEFT, fill="both", expand=True)
        self.scrollbar.grid(row=0, column=1, sticky='ns')
        self.scrollbar.pack(side=RIGHT, fill=Y, expand=False)

        self.frame2 = Frame(self.masterFrame)
        self.frame2.pack(fill=X)
        self.label = Label(self.frame2, text="First input is your nickname")
        self.label.pack(fill=X)

        self.frame3 = Frame(self.masterFrame)
        self.frame3.pack(fill=X)

        self.frame3_1 = Frame(self.frame3, padx=2, pady=1)
        self.frame3_1.pack(fill=X)

        self.frame3_1 = Frame(self.frame3, padx=2, pady=1)
        self.frame3_1.pack(fill=X)
        self.input = Text(self.frame3_1, width=48, height=4)
        self.input.pack(side=LEFT)
        self.sendButton = Button(self.frame3_1, text="Send", width=6, height=3, command=self.sendMessage)
        self.sendButton.pack()
        self.sendButton.bind("<Return>", self.buttonClicked_sub)
        self.input.bind("<KeyRelease-Return>", self.buttonClicked_sub)
    def logRefresh(self, data):
        self.messageLog.configure(state="normal")
        self.sendButton.pack()
        self.sendButton.bind("<Return>", self.buttonClicked_sub)
        self.input.bind("<KeyRelease-Return>", self.buttonClicked_sub)
    def buttonClicked(self):
        self.sendMessage()
    def buttonClicked_sub(self, event):
        self.sendMessage()
    def sendMessage(self):
        global flag
        message = self.input.get(1.0, END)
        sock.send(message)
        if flag:
            self.label.configure(text="Your Nickname is "+"'"+Message[:message.index("\n")]+"'")
            flag = False
        self.input.delete(1.0, END)

class subApp:
    def __init__(self, Master):
        self.master = Master
        Master.geometry("200x67")
        self.mainFrame = Frame(self.master)
        self.mainFrame.pack(fill=X)

        self.frame1 = Frame(self.mainFrame)
        self.frame1.pack(fill=X)
        self.ipLabel = Label(self.frame1)
        self.ipLabel.configure(text="Server IP", width=10)
        self.ipLabel.pack(side=LEFT)
        self.entry1 = Entry(self.frame1)
        self.entry1.pack(side=LEFT)

        self.frame2 = Frame(self.mainFrame)
        self.frame2.pack(fill=X)
        self.portLabel = Label(self.frame2)
        self.portLabel.configure(text="Port", width=10)
        self.portLabel.pack(side+LEFT)
        self.entry2 = Entry(self.frame2)
        self.entry.pack(side=LEFT)

        self.frame3 = Frame(self.mainFrame)
        self.frame3.pack(fill=X)
        self.button = Button(self.frame3)
        self.button.configure(text = "OK", command=self.submit)
        self.button.pack(fill=X)
    def submit(self):
        global HOST, PORT
        HOST = self.entry1.get()
        PORT = int(self.entry2.get())
        self.master.quit()
        self.master.destroy()

    def interface():
        root.mainloop()
    def network():
        sock.connect((HOST, PORT))
        while True:
            data = sock.recv(1024)
            if data:
                app.logRefresh(data)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    root2 = Tk()
    app2 = subApp(root2)
    root2.mainloop()

    root = Tk()
    app = myApp(root)

    thread1 = threading.Thread(target = interface)
    thread2 = threading.Thread(target = network)

    thread1.start()
    thread2.start()