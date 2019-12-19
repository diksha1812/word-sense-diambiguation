<?php
$text = $_GET['plot'];
//echo $text;
$wsd = array("date", "show","switched","cricket","bat","order","bank","fine","race","dump","left","bear");
$n = count($wsd);
//echo $text;
$new_t = preg_split('/[.]/',$text);
$no_sen = sizeof($new_t);
for($z=1;$z<=$no_sen;$z++){
	$words = explode(" ",$new_t[$z]);
	$sen_size = sizeof($words);
	for($i=0;$i<=$sen_size;$i++){
		$j=0;
		while($j<$n){
			if($words[$i] == $wsd[$j]){
				echo "<a href='3.php?sent={$new_t[$z]}&word={$wsd[$j]}'>$wsd[$j]</a>"; //wsd[$j] and $new_t[$z]
				echo " ";
				$j=6000;}   //if
			else
				{$j++;}	
				}   //while
		if($j!=6000){
			echo $words[$i];
			echo " ";
}     //6000 if
}   //for loop 
} //last for loop
?>
