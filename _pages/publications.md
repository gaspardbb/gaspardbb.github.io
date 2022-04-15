---
layout: page
permalink: /publications/
title: publications
description: Find my publications here. Ask for code or slides by mail!
years: [2022, 2021]
nav: true
---
<!-- _pages/publications.md -->
<div class="publications">

{%- for y in page.years %}
  <h2 class="year">{{y}}</h2>
  {% bibliography -f papers -q @*[year={{y}}]* %}
{% endfor %}

</div>
