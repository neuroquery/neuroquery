<?xml version="1.0" encoding="UTF-8"?>

<xsl:transform version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" >

  <xsl:output method="html" version="1.0" encoding="UTF-8"/>

  <xsl:template match="/">
    <html>
      <head>
        <meta charset="UTF-8"/>
        <title>highlighted text</title>
      </head>
      <body>
        <span>
          <xsl:apply-templates/>
        </span>
      </body>
    </html>
  </xsl:template>

  <xsl:template match="extracted_phrase">
    <span style="background-color: LightBlue;"><xsl:apply-templates /></span>
  </xsl:template>

</xsl:transform>

