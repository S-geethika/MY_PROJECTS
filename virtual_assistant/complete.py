import wx
import wikipedia
import wolframalpha

app_id = "VEJK8T-68K2J2GPXJ"
client = wolframalpha.Client(app_id)

class MyFrame(wx.Frame):
    def __init__(self):
        wx.Frame.__init__(self, None,
            pos=wx.DefaultPosition, size=wx.Size(500, 100),
            style=wx.MINIMIZE_BOX | wx.SYSTEM_MENU | wx.CAPTION |
             wx.CLOSE_BOX | wx.CLIP_CHILDREN,
            title="Virtual Assistant")
        panel = wx.Panel(self)
        my_sizer = wx.BoxSizer(wx.VERTICAL)
        lbl = wx.StaticText(panel,label="Hello I am Python Virtual Assistant. What is your query?")
        #png = wx.Image("/media/geethika/New Volume/PROJECTS/va/va.jpeg", wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        my_sizer.Add(lbl, 1, wx.ALL, 5)
        #wx.StaticBitmap(self, -1, png, (1, 5), (png.GetWidth(), png.GetHeight()))
        self.txt = wx.TextCtrl(panel, style=wx.TE_PROCESS_ENTER,size=(400,50))
        self.txt.SetFocus()
        self.txt.Bind(wx.EVT_TEXT_ENTER, self.OnEnter)
        my_sizer.Add(self.txt, 0, wx.ALL, 1)
        panel.SetSizer(my_sizer)
        self.Show()

    def OnEnter(self, event):

        input = self.txt.GetValue()
        input = input.lower()
        try:
        	res = client.query(input)
        	answer = next(res.results).text
        	print (answer)

       	except:
       		try:
       			input = input.split(' ')
       			input = ' '.join(input[2:])
       			print (wikipedia.summary(input))
       		except:
       			print ("I am sorry.....I don't know...please check the your query format (what is ...) ")


if __name__ == "__main__":
    app = wx.App(True)
    frame = MyFrame()
    app.MainLoop()
