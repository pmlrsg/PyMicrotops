import numpy as np
import pandas as pd
from dateutil.parser import parse
from .read_from_serial import read_microtops_serial
from matplotlib.pyplot import xlabel, ylabel, legend


class Microtops:
    """Loads and processes a data file from the Microtops handheld sun photometer.
    Allows easy plotting, and estimation of AOT at an arbitrary wavelength through
    interpolation with the Angstrom exponent

    File should be in CSV format, as produced by the instrument.

    This module requires:
    * numpy
    * pandas
    * dateutil
    """

    def __init__(self, filename, skiprows=None, data_start="FIELDS:", ending="END."):
        """
        Create an Microtops object from a given Microtops data file
        (in CSV format, as provided by the instrument)

        :param filename: Filename of Microtops data to read
        :return:
        skiprows: None, list or int -> Lines to skip, the header is automatically found from what is left
        header_intro: str -> Some microtops have a variable info before the main columns name just before a keyword, 
        this allows to find them automatically
        ending: str, A string that can be found at the end of file and needs removing as well
        """
        self.filename = filename
        self._load_file(filename, skiprows=skiprows, data_start=data_start, ending=ending)

    @classmethod
    def read_from_serial(self, port, filename, **kwargs):
        """
        Read data from a Microtops attached to the computer via a serial port.

        port: Device to read from (eg. COM3 or /dev/serial0 etc)
        filename: Filename to save the data to
        """
        read_microtops_serial(port, filename, **kwargs)
        return Microtops(filename)

    def _load_file(self, filename, skiprows=None, data_start="FIELDS:",ending="END."):
        """
        filename: Filename of Microtops data to read
        skiprows: None, list or int -> Lines to skip, the header is automatically found from what is left
        header_intro: str -> Some microtops have a variable info before the main columns name just before a keyword, 
        this allows to find them automatically
        ending: str, A string that can be found at the end of file and needs removing as well
        """
        #Will read the text to find automatically the Header and remove extra line
        with open(filename) as f:
            content = f.readlines()
        #python closes it automatically
        content = [x.strip() for x in content]
        
        #All lines to skip will be stored in a list called "to_skip"
        if isinstance(skiprows,int):
            #The user entered manually the number of lines to skip
            to_skip = range(0,skiprows+1)
            #The header is the next line
        elif skiprows == None:
            #Will look automatically for the line just before the Columns title
            if data_start in content:
                skip = content.index(data_start)
                to_skip = range(0,skip+1)
            else:
                to_skip = []
        else:
            if not isinstance(skiprows,list):
                print("Skiprows type not handled: {}".format(type(skiprows)))
                raise TypeError("Skiprows type not handled: {}".format(type(skiprows)))
        
        #Now will look if the document has an ending that needs stripping as well
        if content[-1] == ending:
            end_index = int(len(content)-1)
        else:
            end_index = None

        if end_index is not None and end_index not in to_skip:
            to_skip.append(end_index)
        
        #Will make a list to remove  NaN entries as well
        nan_lines = []
        for i in range(0, len(content)):
            #each line is a str
            if "Na" in content[i]:
                nan_lines.append(i)
        if nan_lines != []:
            #add that to the lines to skip
            to_skip = to_skip+nan_lines

        self.data = pd.read_csv(filename,skiprows=to_skip)

        def f(s):
            return parse(s['DATE'] + " " + s['TIME'])

        self.data.index = pd.DatetimeIndex(self.data.apply(f, axis=1))

        self._process_wavelengths()

    def plot(self, wavelengths=None, start_time=None, end_time=None, **kwargs):
        """
        Plot the AOT data, with an optional set of wavelengths (as a list), and
        a start and end time as strings.
        """
        data = self.data[start_time:end_time]

        if wavelengths is None:
            wavelengths = self.wavelengths

        col_names = ['AOT%d' % (int(x)) for x in wavelengths]

        data.ix[:, col_names].plot(**kwargs)
        legend(loc='best')
        xlabel('Time')
        ylabel('AOT')

    def _process_wavelengths(self):
        """
        Extract wavelengths from the column headers
        """
        aot_cols = [c for c in self.data.columns if 'AOT' in c]
        wvs = [int(x.replace('AOT', '')) for x in aot_cols]

        self.wavelengths = wvs

    def aot(self, wavelength, start_time=None, end_time=None):
        """
        Get AOT at a given wavelength.

        Returned as a pandas Series over the range of start_time to end_time.

        :param wavelength: Wavelength at which AOT should be retrieved (in nm)
        :param start_time: Start time for temporal subsetting (as a string in yyyy-mm-dd hh:mm:ss)
        :param end_time: End time for temporal subsetting (as a string in yyyy-mm-dd hh:mm:ss)
        :return:
        """
        data = self.data[start_time:end_time]

        wavelength = int(wavelength)

        if wavelength in self.wavelengths:
            # This wavelength was measured by the Microtops,
            # so just return it
            return data['AOT%d' % wavelength]
        else:
            # Need to interpolate using Angstrom exp

            # First we choose the two closest wavelengths
            wvs = np.array(self.wavelengths)
            diff = wavelength - wvs
            try:
                wv_below = wvs[np.argmin(diff[diff > 0])]
            except ValueError:
                # If the above line of code gave an error then we
                # are dealing with a wavelength lower than the minimum wavelength
                # therefore we will be extrapolating, so print a warning
                print("Warning: extrapolating using Angstrom coefficient")
                # and then use the lowest wavelength we have
                wv_below = wvs[0]
            try:
                wv_above = wvs[np.argmin(diff[diff < 0])]
            except ValueError:
                # If the above line of code gave an error then we
                # are dealing with a wavelength higher than the maximum wavelength
                # therefore we will be extrapolating, so print a warning
                print("Warning: extrapolating using Angstrom coefficient")
                # and then use the lowest wavelength we have
                wv_below = wvs[-1]

            aot_below = data["AOT%d" % wv_below]
            aot_above = data["AOT%d" % wv_above]

            # Then we calculate the Angstrom exp for every observation
            angstrom = -1 * (np.log(aot_below / aot_above) / (np.log(float(wv_below) / wv_above)))

            # Then we use the exponent to interpolate
            result = aot_below * ((float(wavelength) / wv_below) ** (-1 * angstrom))

            return result
