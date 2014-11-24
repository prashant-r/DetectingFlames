package detectingFlames;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.regex.Matcher;
import java.util.regex.Pattern;



public class starter {
	public static void main(String args[]) throws IOException
	{
		System.out.println("reaady to rockk and roll!");
		starter obj = new starter();
		obj.readCSV();
	 
	}
	public static int indexFrom(String s,char chars, int lastIndex)
    {
        for (int i=0;i<s.length();i++)
           if (s.charAt(i) == chars && i>lastIndex)
              return i;
        return -1;
    } 
	 public void readCSV() throws IOException {
		 	File currentDirectory = new File(new File(".").getAbsolutePath());
		 	System.out.println(currentDirectory.getCanonicalPath());
			String csvFile = currentDirectory.getCanonicalPath()+ "/DataFiles/train.csv";
			BufferedReader br = null;
			String line = "";
			char csvSplitBy = ',';
			FileWriter writer = new FileWriter(currentDirectory.getCanonicalPath()+ "/QAFiles/trainPreProcess.txt");
			String comments[] = new String[3];
			
			try {
		 
				br = new BufferedReader(new FileReader(csvFile));
				int c =0;
				while ((line = br.readLine()) != null) {
//					if(c++ <5)
//					{
//						continue;
//					}
//					else if(c++ >6)
//					{
//						break;
//					}
//					
						int r = 0;	
						int commaIndex =0;
						int firstIndex =0;
						String s = "";
						while(r<2)
						{
							commaIndex = starter.indexFrom(line, csvSplitBy, commaIndex);
							String toSave = line.substring(firstIndex,commaIndex);
							firstIndex = commaIndex+1;
							comments[r] = toSave;
							r++;
						}
					comments[2] =line.substring(firstIndex,line.length());
					parseThroughComment(comments[2], writer);
					}
			writer.flush();
		    writer.close();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			} catch (IOException e) {
				e.printStackTrace();
			} finally {
				if (br != null) {
					try {
						br.close();
					} catch (IOException e) {
						e.printStackTrace();
					}
				}
			}
		 
			System.out.println("Done");
		  }

	public void parseThroughComment(String comment, FileWriter fileWriter) throws IOException
	{	
		try
		{
		    //generate whatever data you want
			comment = preProcessComment(comment);
			String[] words = comment.split(" ");   
			for (String word : words)  
			{  
			   System.out.println(word);  
			}  
			fileWriter.write(comment +  System.getProperty("line.separator"));
		    
		}
		catch(IOException e)
		{
		     e.printStackTrace();
		} 
	}
	private String preProcessComment(String comment) {
		comment = comment.replace('_', ' ');
		comment = comment.replace("\\\\", "\\");
		comment = comment.replaceAll("\\\\n", "");
		comment = comment.replaceAll("\\\\r", "");
		comment = comment.replaceAll("\\\\'", "'");
		comment = comment.replaceAll("\\\\\\\\'", "'");
		comment = comment.replaceAll("\\\\\\\\","");
		comment = this.removeUTFCharacters(comment);
		comment = comment.replaceAll("\\\\t","");
		comment = comment.replaceAll("\\\\", "");
		comment = this.removeUnicodeCharacters(comment);
		return comment;
	}
	
	public String removeUTFCharacters(String data){
		Pattern p = Pattern.compile("\\\\x(\\p{XDigit}{2})");
		Matcher m = p.matcher(data);
		StringBuffer buf = new StringBuffer(data.length());
		while (m.find()) {
		String ch = String.valueOf((char) Integer.parseInt(m.group(1), 16));
		m.appendReplacement(buf, Matcher.quoteReplacement(ch));
		}
		m.appendTail(buf);
		return buf.toString();
		}
	public String removeUnicodeCharacters(String data){
		Pattern p = Pattern.compile("\\\\u(\\p{XDigit}{4})");
		Matcher m = p.matcher(data);
		StringBuffer buf = new StringBuffer(data.length());
		while (m.find()) {
		String ch = String.valueOf((char) Integer.parseInt(m.group(1), 16));
		m.appendReplacement(buf, Matcher.quoteReplacement(ch));
		}
		m.appendTail(buf);
		return buf.toString();
		}

}
