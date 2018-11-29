import ctypes
import os
import platform
import sys

#https://stackoverflow.com/questions/51658/cross-platform-space-remaining-on-volume-using-python

def get_free_space_mb(dirname):
    """Return folder/drive free space (in megabytes)."""
    if platform.system() == 'Windows':
        free_bytes = ctypes.c_ulonglong(0)
        ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p(dirname), None, None, ctypes.pointer(free_bytes))
        return free_bytes.value / 1024 / 1024
    else:
        st = os.statvfs(dirname)
        return st.f_bavail * st.f_frsize / 1024 / 1024


print("get_free_space_mb('c:/')",get_free_space_mb("c:/"))
print("get_free_space_mb('d:/')",get_free_space_mb("d:/"))