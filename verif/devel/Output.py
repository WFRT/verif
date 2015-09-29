   def _savePlot(self, data):
      if(self._figsize is not None):
         mpl.gcf().set_size_inches(int(self._figsize[0]), int(self._figsize[1]))
      if(not self._showMargin):
         Common.removeMargin()
      if(self._filename is not None):
         extension = self._filename.split('.')[-1]
         if(extension == "html"):
            import bokeh.mpl
            from bokeh.plotting import output_file, show
            output_file("test.html")
            show(bokeh.mpl.to_bokeh())
         else:
            mpl.savefig(self._filename, bbox_inches='tight', dpi=self._dpi)
      else:
         fig = mpl.gcf()
         fig.canvas.set_window_title(data.getFilenames()[0])
         mpl.show()
