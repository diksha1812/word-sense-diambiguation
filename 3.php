<?php
$line = $_GET['sent'];
$w = $_GET['word'];
#echo $line;
//echo $w;
$result = system("python3 final.py .$line .$w");
echo $result;
echo "done";
?>
