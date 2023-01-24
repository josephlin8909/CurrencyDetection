package edu.purdue.tada_chat_mom.networks;

import android.content.Intent;
import android.content.pm.ActivityInfo;
import android.graphics.PixelFormat;
import android.os.AsyncTask;
import android.os.Bundle;
import android.provider.Settings;
import android.util.Log;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import javax.net.ssl.HostnameVerifier;
import javax.net.ssl.HttpsURLConnection;

import edu.purdue.tada_chat_mom.R;
import edu.purdue.tada_chat_mom.utils.ActivityBridge;
import edu.purdue.tada_chat_mom.utils.PreferenceHelper;

public class HttpsRecTag extends TadaUtils {
	/*
	 * HttpSendImage is an ACTIVITY, as it is part of the User Interface (UI) -
	 * it opens a new layout view and shows the uploaded picture and relevant
	 * data
	 */

	/* Specify server-side filename */
	private String PHP_FILENAME = PATH + "rest_get_results.php";

	/* Create tag for logcat */
	private static final String TAG = "YU-HTTPRECTAG";

	private String imageName = null;
	private String imgFilePath = null;
	/* Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		getWindow().setFormat(PixelFormat.TRANSLUCENT);
	    requestWindowFeature(Window.FEATURE_NO_TITLE);
	    getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
	    setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);
		/* Insert code to setup activity layout ... */
	    setContentView(R.layout.connection_layout);
		/*
		 * Create async task to receive tag files from the server while showing a
		 * progress dialog
		 */
		
	    ProgressBar progressBar = (ProgressBar)findViewById(R.id.progressBar_tag);
	    TextView loadingText = (TextView)findViewById(R.id.loading_text);
	    loadingText.setText("Loading...");
		DownloadDataTask ddt = new DownloadDataTask(progressBar);
		ddt.execute();
	}

	/**
	 * Extends {@link AsyncTask}. Shows progress bar while receiving the tag file from
	 * the server
	 * 
	 * @author Yu Wang
	 * 
	 */
	public class DownloadDataTask extends AsyncTask<Void, Integer, Void> {
		
		ProgressBar pb;
		public DownloadDataTask(ProgressBar progressBar){
			pb = progressBar;
		}
		
		

		// can use UI thread here
		protected void onPreExecute() {
			//dialog.setMessage("Refreshing...");
			//dialog.setCancelable(false);
			//dialog.show();
		}

		// automatically done on worker thread (separate from UI thread)
		protected Void doInBackground(final Void... args) {

			/*
			 * Get image filepath - I use a singleton (ActivityBridge) to set
			 * and get parameters - You have to decide how you are going to
			 * access your variables given your specific needs
			 */

			try {
				/* Send image to server */
				Log.d(TAG, "Sendin image to server...");

				/*
				 * Send image and save response in a string to put in a TextView
				 * (GUI)
				 */

				/* Get user ID and password (MD5) */
				String userIdLogin = PreferenceHelper.getUserId(getBaseContext());
				System.out.println("userID:"+ userIdLogin);
				String pwdMd5Login = "tadaapi298";
				String deviceID = Settings.Secure.getString(getContentResolver(),
						Settings.Secure.ANDROID_ID);
				imgFilePath = getIntent().getStringExtra("firstEntry");
				System.out.println("deviceID:" + deviceID);
				imageName = imgFilePath.substring(imgFilePath.lastIndexOf("/") + 1);
				Log.d(TAG, imageName);
				/*
				 * Call method to create HTTPS connection and communicate with
				 * server
				 */
				String response = sendImageAndLogin(imageName,deviceID, PHP_FILENAME,
						userIdLogin, pwdMd5Login);

				/* Set response to singleton to analyze it on onPostExecute */
				ActivityBridge.getInstance().setHttpsresponse(response);

			} catch (IOException e) {
				e.printStackTrace();
			}

			return null;
		}

		// can use UI thread here
		protected void onPostExecute(Void unused) {
			/*if (this.dialog.isShowing()) {
				this.dialog.dismiss();
			}*/

			/* Get bitmap of current image (uploaded image) from singleton */
			//Bitmap rgbBitmap = ActivityBridge.getInstance().getRgbBitmap();

			/* Set bitmap of current image (uploaded image) to singleton */
			//ActivityBridge.getInstance().setScreenSizeBitmap(rgbBitmap);

			/* Get HTTPS response from singleton */
			String response = ActivityBridge.getInstance().getHttpsresponse();
			System.out.println("response::"+ response);
			/* Check server response */
			if (response.compareTo("no results yet\n") == 0) {

				/* Show dialog to tell user something went wrong */
				System.out.println("No tag file yet");//TODO: may have to use Dialog later!
				Toast.makeText(getApplicationContext(), "No tag file yet", Toast.LENGTH_LONG);
				Log.d(TAG, "I am in httpsrectag") ;
				HttpsRecTag.this.setResult(RESULT_CANCELED);
				finish();
			} else {
				
				/* Set image and text to layout */
				
				response = response.substring(response.indexOf("\n")+1);
				System.out.println("response::"+ response);
				Log.d(TAG, "Response is " + response) ;
				String tagFilePath = imgFilePath.substring(0, imgFilePath.lastIndexOf("."))+".tag";
				
				try{
					File tagFile = new File(tagFilePath);
					FileOutputStream fos = new FileOutputStream(tagFile,false);
					
					//deleteFile(tagFilePath);
					if (!tagFile.exists()) {
						tagFile.createNewFile();	
					}
					fos.write((response).getBytes()); 
					fos.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
		       Intent intent = new Intent();
		       intent.putExtra("tag_file", tagFilePath);
				HttpsRecTag.this.setResult(RESULT_OK,intent);
				finish();
			}
			
		}
	}

	private String sendImageAndLogin(String imageName,String deviceID, String serverFilename,
			String userIdLogin, String pwdMd5Login) throws IOException {

		/* Always verify the host - Do not check for certificate */
		HostnameVerifier v = TadaUtils.hostVerify();

		/* Get server URL from singleton */
		//String server = ActivityBridge.getInstance().getServerDomainName();
		String server = SERVER_NAME;
		
		/* Open HTPPS connection to the server */
		HttpsURLConnection httpsConnection = getHttpsConnection(serverFilename,
				v, server, "multipart/form-data;boundary=" + BOUNDARY);

		/* Create output stream to write bytes (headers/data/image) */
		DataOutputStream outputStream = createOutputStream(httpsConnection);

		String httpsResponseBody;

		/* Wrong server request */
		if (outputStream == null) {
			/* We need to check this on the returning method */
			httpsResponseBody = null;

		} else {

			/* Send data/headers */
			outputStream = outputStreamHeaderMultiPartForm(outputStream);

			// --------------- LOGIN ---------------

			/* Set up output stream for user ID form */
			outputStream = outputStreamSetUpForm(outputStream, PHP_USER_ID,
					userIdLogin);

			/* Set up output stream for MD5 password form */
			outputStream = outputStreamSetUpForm(outputStream, PHP_PWD,
					pwdMd5Login);
			
			/*Information of the first image */
			outputStream = outputStreamSetUpForm(outputStream, "deviceID",
					deviceID);	
			outputStream = outputStreamSetUpForm(outputStream, "imageName",
					imageName);
			
			
			//End of the first image
			
			
			
			// -------------------------------------

			/* Send multipart form data necessary after file data */
			outputStreamComplete(outputStream);

			/* Print code and message to the log */
			getServerResponse(httpsConnection);

			/* Get data from server and convert it to string */
			InputStream httpsInputStream = httpsConnection.getInputStream();
			httpsResponseBody = convertStreamToString(httpsInputStream);

			Log.d(TAG, "Response Body: " + httpsResponseBody);

			/* Close streams */
			TadaUtils.closeStreams(httpsInputStream, outputStream);
			

			/* Parse received data */
		}

		return httpsResponseBody;
	}
}




######Some function called in this part of code#######
	public final static HttpsURLConnection getHttpsConnection(String filename,
			HostnameVerifier v, String server, String contentType)
			throws IOException {

		HttpsURLConnection httpsConnection = null;

		trustAllHosts();

		Log.d(TAG, "Opening HTTPS Connection");

		URL url = new URL("https", server, filename);

		httpsConnection = (HttpsURLConnection) url.openConnection();

		/* Use a POST method */
		httpsConnection.setRequestMethod("POST");

		httpsConnection.setHostnameVerifier(v);

		/* Allow inputs */
		httpsConnection.setDoInput(true);

		/* Allow outputs */
		httpsConnection.setDoOutput(true);

		/* Don't use a cached copy */
		httpsConnection.setUseCaches(false);

		httpsConnection.setRequestProperty("Connection", "Keep-Alive");

		/* Set content type */
		httpsConnection.setRequestProperty("Content-Type", contentType);

		Log.d(TAG, "Initialized HTTPS Connection");

		Log.d(TAG, "httpsConnection: " + httpsConnection);

		return httpsConnection;
	}


	public static DataOutputStream createOutputStream(
			HttpsURLConnection httpsConnection) {

		DataOutputStream outputStream = null;

		try {
			outputStream = new DataOutputStream(
					httpsConnection.getOutputStream());

		} catch (IOException e) {
			/*
			 * If tehre's something wrong with the server, catching the
			 * exception will set the output stream to null
			 */
			Log.d(TAG, "IOException: " + e.toString());
		}

		return outputStream;
	}



	public final static DataOutputStream outputStreamSetUpForm(
			DataOutputStream outputStream, String key, String value)
			throws IOException {

		Log.d(TAG, "Starting headers for form");

		outputStream.writeBytes("Content-Disposition: form-data; name=\"" + key
				+ "\"" + LINE_END + LINE_END + value + LINE_END + TWO_HYPHENS
				+ BOUNDARY + LINE_END);

		Log.d(TAG, "Headers for form are written");

		return outputStream;
	}