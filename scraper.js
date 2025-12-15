const cheerio = require("cheerio");
const fs = require("fs");

const AUCTION_URL =
  "https://www.espncricinfo.com/auction/ipl-2026-auction-1515016";

async function getHtml(url) {
  return await fetch(url).then((response) => response.text());
}

playerTypes = {
  BAT: "Batter",
  AR: "All Rounder",
  BOWL: "Bowler",
};

async function getTeamAuctionUrls(teamName) {
  const html = await getHtml(AUCTION_URL);
  const $ = cheerio.load(html);
  const links = $("a");
  for (const link of links) {
    if (link.attribs.href.includes(teamName)) {
      return link.attribs.href.split("/").pop();
    }
  }
  return null;
}

async function getTeamSquad(teamName) {
  const url = await getTeamAuctionUrls(teamName);
  if (!url) {
    return null;
  }
  const html = await getHtml(AUCTION_URL + "/" + url);
  const $ = cheerio.load(html);
  const squadTable = $("table tbody");
  const rows = squadTable.find("tr");
  const squad = [];
  for (const row of rows) {
    const cells = $(row).find("td");
    const playerName = cells.eq(0).text();
    const isOverseas = cells
      .eq(0)
      .find("i")
      .hasClass("icon-airplanemode_active-filled");

    const playingType = playerTypes[cells.eq(1).text()];
    const playerPrice = Number(cells.eq(3).text()) * 100;

    squad.push({
      playerName,
      isOverseas,
      playingType,
      playerPrice,
    });
  }

  return squad;
}

function getTeams() {
  const teams = fs.readFileSync("teams.json", "utf8");
  return JSON.parse(teams);
}

async function main() {
  const teams = await getTeams();
  for (const team of teams) {
    teamName = team.toLowerCase().replaceAll(" ", "-");
    const squad = await getTeamSquad(teamName);
    fs.writeFileSync(`squads/${teamName}.json`, JSON.stringify(squad, null, 2));
  }
}

main().catch((error) => {
  if (error) console.error(error);
});
