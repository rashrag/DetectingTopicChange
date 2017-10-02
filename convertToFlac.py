import pydub

def convertToFlac(audioFileName):
    pydub.AudioSegment.converter = r"ffmpeg-20170921-183fd30-win64-static\bin\ffmpeg.exe"
    song = pydub.AudioSegment.from_wav(audioFileName)
    exportName = audioFileName +".flac"
    song.export(exportName, format="flac")

if __name__=="__main__":
    convertToFlac("test.wav")