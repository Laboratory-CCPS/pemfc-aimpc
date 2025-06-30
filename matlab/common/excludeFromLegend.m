% excludeFromLegend
%
%	Legt fest, dass für das angegebene Line-Objekt kein Legendeneintrag
%	erzeugt wird. Damit kann erreicht werden, dass z.B. Hilfslinien etc.
%	nicht in der Legende auftauchen.
%
%	Syntax
%		excludeFromLegend(hLine)
%
%	Argumente
%		hLine	Handle (oder Vektor mit Handles) zu Line-Objekt
%
%	Rückgabewerte
%		keine
%
%	Beispiel
%		plot((0:0.01:10)', sin(0:0.01:10)');
%		hold('on');
%		hZero = plot([0; 10], [0; 0], 'g');
%		excludeFromLegend(hZero);
%		legend('sin');

%	Versionen
%		1.0.1	14.01.2011	EL
%				Erweiterung auf Vektoren
%
%		1.0.0	13.01.2011	EL

function excludeFromLegend(hLine)

	for l = 1:length(hLine)
		hAnnotation = get(hLine(l), 'Annotation');
		hLegendEntry = get(hAnnotation', 'LegendInformation');
		set(hLegendEntry, 'IconDisplayStyle', 'off');
	end % for l
	
end % function excludeFromLegend
