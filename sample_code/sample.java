//----------------first image ------------
/* Set up output stream for file upload */
outputStream = TadaUtils.outputStreamSetUpFile(imagePath1, outputStream, "file");
//outputStream = outputStreamSetUpFile2(filepath2,imagePath, outputStream, "file","file2");

/* Create input stream to put the image */
FileInputStream fileInputStream = new FileInputStream(new File(imagePath1));

/* Retrieve file and write it to output stream to be sent */
TadaUtils.retrieveFileBytes(imagePath1, outputStream, fileInputStream);
//End of the first image



public final static String LINE_END = "\n"; /*
                                             * NOTE: USED TO BE \r\n -
                                             * Windows Line
                                             */
public final static String TWO_HYPHENS = "--";
public final static String BOUNDARY = "*****";



public final static DataOutputStream outputStreamHeaderMultiPartForm(
        DataOutputStream outputStream) throws IOException {

    Log.d(TAG, "Starting headers for multi-part form");

    /* Send data/headers */
    outputStream.writeBytes(TWO_HYPHENS + BOUNDARY + LINE_END);

    Log.d(TAG, "Headers for multi-part form are written");

    return outputStream;
}


public final static DataOutputStream outputStreamSetUpFile(String filepath,
        DataOutputStream outputStream, String tag) throws IOException {

    Log.d(TAG, "Starting headers for file upload");

    outputStream
            .writeBytes("Content-Disposition: form-data; name=\"" + tag + "\";filename=\""
                    + filepath + "\"" + LINE_END);

    outputStream.writeBytes(LINE_END);

    Log.d(TAG, "Headers for file upload are written");

    return outputStream;
}


public final static DataOutputStream retrieveFileBytes(String filepath,
        DataOutputStream outputStream, FileInputStream fileInputStream)
        throws IOException {

    int bytesRead, bytesAvailable, bufferSize;
    byte[] buffer;
    int maxBufferSize = 1 * 1024 * 1024;

    bytesAvailable = fileInputStream.available();

    bufferSize = Math.min(bytesAvailable, maxBufferSize);

    buffer = new byte[bufferSize];

    /* Read file (image) */
    bytesRead = fileInputStream.read(buffer, 0, bufferSize);

    /* Send image */
    while (bytesRead > 0) {
        outputStream.write(buffer, 0, bufferSize);
        bytesAvailable = fileInputStream.available();
        bufferSize = Math.min(bytesAvailable, maxBufferSize);
        bytesRead = fileInputStream.read(buffer, 0, bufferSize);
    }

    return outputStream;
}