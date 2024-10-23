# understanding_pickle_files

Python pickle module is used for serializing and de-serializing a Python object structure. Any object in Python can be pickled so that it can be saved on disk. What Pickle does is it “serializes” the object first before writing it to a file.
Pickling is a way to convert a Python object (list, dictionary, etc.) into a character stream.
The idea is that this character stream contains all the information necessary to reconstruct the object in another Python script. It provides a facility to convert any Python object to a byte stream. 
This Byte stream contains all essential information about the object so that it can be reconstructed, or “unpickled” and get back into its original form in any Python.

